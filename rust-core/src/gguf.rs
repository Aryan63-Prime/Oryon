//! Zero-dependency GGUF v2/v3 loader supporting F32, F16, Q8_0, and Q4_0 tensors
//! plus embedded tokenizer vocabulary for Llama-3 / Qwen2.5 / SmolLM2 models.

use crate::config::{ModelConfig, QuantType};
use crate::model::{LayerWeights, Transformer};
use crate::quant::{WeightMatrix, BLOCK_SIZE};
use crate::tokenizer::BpeTokenizer;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::Path;

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const GGML_TYPE_F32: u32 = 0;
const GGML_TYPE_F16: u32 = 1;
const GGML_TYPE_Q4_0: u32 = 2;
const GGML_TYPE_Q8_0: u32 = 8;

enum GgufVal {
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    StrArray(Vec<String>),
    Other,
}

struct TensorInfo {
    dims: Vec<usize>,
    ggml_type: u32,
    offset: u64,
}

pub fn load_gguf<P: AsRef<Path>>(path: P, max_seq_cap: usize) -> io::Result<(Transformer, BpeTokenizer)> {
    let mut r = BufReader::new(File::open(path)?);
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != GGUF_MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Not a GGUF file"));
    }
    let version = read_u32(&mut r)?;
    if version < 2 || version > 3 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unsupported GGUF version: {}", version),
        ));
    }
    let tensor_count = read_u64(&mut r)? as usize;
    let kv_count = read_u64(&mut r)? as usize;

    let mut meta: HashMap<String, GgufVal> = HashMap::new();
    for _ in 0..kv_count {
        let key = read_gguf_string(&mut r)?;
        let vtype = read_u32(&mut r)?;
        let val = read_gguf_val(&mut r, vtype)?;
        meta.insert(key, val);
    }

    let mut tensors: HashMap<String, TensorInfo> = HashMap::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let name = read_gguf_string(&mut r)?;
        let n_dims = read_u32(&mut r)? as usize;
        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dims.push(read_u64(&mut r)? as usize);
        }
        let ggml_type = read_u32(&mut r)?;
        let offset = read_u64(&mut r)?;
        tensors.insert(name, TensorInfo { dims, ggml_type, offset });
    }

    let alignment = get_u32(&meta, "general.alignment").unwrap_or(32) as u64;
    let cur_pos = r.stream_position()?;
    let data_start = ((cur_pos + alignment - 1) / alignment) * alignment;

    let arch = get_str(&meta, "general.architecture").unwrap_or("llama").to_string();
    let dim = get_u32(&meta, &format!("{}.embedding_length", arch))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing embedding_length"))? as usize;
    let hidden_dim = get_u32(&meta, &format!("{}.feed_forward_length", arch))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing feed_forward_length"))? as usize;
    let n_layers = get_u32(&meta, &format!("{}.block_count", arch))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing block_count"))? as usize;
    let n_heads = get_u32(&meta, &format!("{}.attention.head_count", arch))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing head_count"))? as usize;
    let n_kv_heads = get_u32(&meta, &format!("{}.attention.head_count_kv", arch))
        .unwrap_or(n_heads as u32) as usize;
    let rope_theta = get_f32(&meta, &format!("{}.rope.freq_base", arch)).unwrap_or(10000.0);
    let rms_eps = get_f32(&meta, &format!("{}.attention.layer_norm_rms_epsilon", arch)).unwrap_or(1e-5);
    let ctx_len = get_u32(&meta, &format!("{}.context_length", arch)).unwrap_or(2048) as usize;
    let max_seq_len = ctx_len.min(max_seq_cap).max(256);

    let vocab_tokens: Vec<Vec<u8>> = match meta.get("tokenizer.ggml.tokens") {
        Some(GgufVal::StrArray(arr)) => arr.iter().map(|s| s.as_bytes().to_vec()).collect(),
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "GGUF missing tokenizer.ggml.tokens",
            ))
        }
    };
    let vocab_size = vocab_tokens.len();
    let bos_id = get_u32(&meta, "tokenizer.ggml.bos_token_id").unwrap_or(1);
    let eos_id = get_u32(&meta, "tokenizer.ggml.eos_token_id").unwrap_or(2);
    let add_bos = get_bool(&meta, "tokenizer.ggml.add_bos_token").unwrap_or(true);
    let tokenizer = BpeTokenizer::from_gguf_vocab(vocab_tokens, bos_id, eos_id);

    let token_embedding = read_weight_matrix(&mut r, &tensors, data_start, "token_embd.weight")?;
    let primary_quant = match &token_embedding {
        WeightMatrix::Q4_0 { .. } => QuantType::Q4_0,
        WeightMatrix::Q8_0 { .. } => QuantType::Q8_0,
        WeightMatrix::F32 { .. } => QuantType::F32,
    };

    let mut layers = Vec::with_capacity(n_layers);
    for l in 0..n_layers {
        let p = format!("blk.{}", l);
        layers.push(LayerWeights {
            attn_norm: read_norm_vec(&mut r, &tensors, data_start, &format!("{}.attn_norm.weight", p))?,
            wq: read_weight_matrix(&mut r, &tensors, data_start, &format!("{}.attn_q.weight", p))?,
            wk: read_weight_matrix(&mut r, &tensors, data_start, &format!("{}.attn_k.weight", p))?,
            wv: read_weight_matrix(&mut r, &tensors, data_start, &format!("{}.attn_v.weight", p))?,
            wo: read_weight_matrix(&mut r, &tensors, data_start, &format!("{}.attn_output.weight", p))?,
            ffn_norm: read_norm_vec(&mut r, &tensors, data_start, &format!("{}.ffn_norm.weight", p))?,
            w_gate: read_weight_matrix(&mut r, &tensors, data_start, &format!("{}.ffn_gate.weight", p))?,
            w_up: read_weight_matrix(&mut r, &tensors, data_start, &format!("{}.ffn_up.weight", p))?,
            w_down: read_weight_matrix(&mut r, &tensors, data_start, &format!("{}.ffn_down.weight", p))?,
        });
    }

    let final_norm = read_norm_vec(&mut r, &tensors, data_start, "output_norm.weight")?;
    let lm_head = if tensors.contains_key("output.weight") {
        Some(read_weight_matrix(&mut r, &tensors, data_start, "output.weight")?)
    } else {
        None
    };

    let config = ModelConfig {
        quant_type: primary_quant,
        vocab_size,
        dim,
        hidden_dim,
        n_layers,
        n_heads,
        n_kv_heads,
        max_seq_len,
        rope_theta,
        rms_eps,
        tied_embeddings: lm_head.is_none(),
        add_bos,
    };
    config.validate()?;

    Ok((
        Transformer {
            config,
            token_embedding,
            layers,
            final_norm,
            lm_head,
        },
        tokenizer,
    ))
}

fn read_norm_vec<R: Read + Seek>(
    r: &mut R,
    tensors: &HashMap<String, TensorInfo>,
    data_start: u64,
    name: &str,
) -> io::Result<Vec<f32>> {
    let info = tensors.get(name).ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, format!("Missing tensor: {}", name))
    })?;
    let n: usize = info.dims.iter().product();
    r.seek(SeekFrom::Start(data_start + info.offset))?;
    match info.ggml_type {
        GGML_TYPE_F32 => {
            let mut raw = vec![0u8; n * 4];
            r.read_exact(&mut raw)?;
            Ok(raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect())
        }
        GGML_TYPE_F16 => {
            let mut raw = vec![0u8; n * 2];
            r.read_exact(&mut raw)?;
            Ok(raw
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
                .collect())
        }
        t => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unsupported norm tensor type {} for {}", t, name),
        )),
    }
}

fn read_weight_matrix<R: Read + Seek>(
    r: &mut R,
    tensors: &HashMap<String, TensorInfo>,
    data_start: u64,
    name: &str,
) -> io::Result<WeightMatrix> {
    let info = tensors.get(name).ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, format!("Missing tensor: {}", name))
    })?;
    if info.dims.len() != 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Expected 2D tensor for {}", name),
        ));
    }
    // GGUF stores dims in column-major order: dims[0] = cols (inner), dims[1] = rows (outer)
    let cols = info.dims[0];
    let rows = info.dims[1];
    let total = rows * cols;
    r.seek(SeekFrom::Start(data_start + info.offset))?;

    match info.ggml_type {
        GGML_TYPE_F32 => {
            let mut raw = vec![0u8; total * 4];
            r.read_exact(&mut raw)?;
            let data = raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            Ok(WeightMatrix::F32 { rows, cols, data })
        }
        GGML_TYPE_F16 => {
            let mut raw = vec![0u8; total * 2];
            r.read_exact(&mut raw)?;
            let data = raw
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
                .collect();
            Ok(WeightMatrix::F32 { rows, cols, data })
        }
        GGML_TYPE_Q8_0 => {
            // GGUF Q8_0 block: 2-byte f16 scale + 32 x i8 = 34 bytes
            let n_blocks = total / BLOCK_SIZE;
            let mut scales = Vec::with_capacity(n_blocks);
            let mut quants = Vec::with_capacity(total);
            let mut buf = [0u8; 2 + BLOCK_SIZE];
            for _ in 0..n_blocks {
                r.read_exact(&mut buf)?;
                let d = f16_to_f32(u16::from_le_bytes([buf[0], buf[1]]));
                scales.push(d);
                for &b in &buf[2..] {
                    quants.push(b as i8);
                }
            }
            Ok(WeightMatrix::Q8_0 {
                rows,
                cols,
                scales,
                quants,
            })
        }
        GGML_TYPE_Q4_0 => {
            // GGUF Q4_0 block: 2-byte f16 scale + 16 packed bytes = 18 bytes.
            // Note: GGUF stores nibbles split-half (low nibbles = 0..16, high nibbles = 16..32),
            // so we repack into our interleaved even/odd layout for fast SIMD matvec.
            let n_blocks = total / BLOCK_SIZE;
            let mut scales = Vec::with_capacity(n_blocks);
            let mut packed = Vec::with_capacity(total / 2);
            let mut buf = [0u8; 2 + BLOCK_SIZE / 2];
            for _ in 0..n_blocks {
                r.read_exact(&mut buf)?;
                let d = f16_to_f32(u16::from_le_bytes([buf[0], buf[1]]));
                scales.push(d);
                let mut unp = [0u8; BLOCK_SIZE];
                for i in 0..16 {
                    let b = buf[2 + i];
                    unp[i] = b & 0x0F;
                    unp[i + 16] = (b >> 4) & 0x0F;
                }
                for i in 0..16 {
                    packed.push((unp[2 * i] & 0x0F) | ((unp[2 * i + 1] & 0x0F) << 4));
                }
            }
            Ok(WeightMatrix::Q4_0 {
                rows,
                cols,
                scales,
                packed,
            })
        }
        t => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "GGUF quant type {} for {} not yet supported (use Q4_0, Q8_0, F16, or F32 GGUF)",
                t, name
            ),
        )),
    }
}

#[inline]
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1F) as u32;
    let frac = (h & 0x03FF) as u32;
    let bits = if exp == 0 {
        if frac == 0 {
            sign
        } else {
            let mut f = frac;
            let mut e: i32 = -14;
            while (f & 0x0400) == 0 {
                f <<= 1;
                e -= 1;
            }
            f &= 0x03FF;
            sign | (((e + 127) as u32) << 23) | (f << 13)
        }
    } else if exp == 31 {
        sign | (0xFF << 23) | (frac << 13)
    } else {
        sign | ((exp + 112) << 23) | (frac << 13)
    };
    f32::from_bits(bits)
}

fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_gguf_string<R: Read>(r: &mut R) -> io::Result<String> {
    let len = read_u64(&mut *r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

fn read_gguf_val<R: Read>(r: &mut R, vtype: u32) -> io::Result<GgufVal> {
    match vtype {
        0 | 1 => {
            let mut b = [0u8; 1];
            r.read_exact(&mut b)?;
            Ok(GgufVal::Other)
        }
        2 | 3 => {
            let mut b = [0u8; 2];
            r.read_exact(&mut b)?;
            Ok(GgufVal::Other)
        }
        4 => Ok(GgufVal::U32(read_u32(r)?)),
        5 => Ok(GgufVal::I32(read_u32(r)? as i32)),
        6 => Ok(GgufVal::F32(f32::from_bits(read_u32(r)?))),
        7 => {
            let mut b = [0u8; 1];
            r.read_exact(&mut b)?;
            Ok(GgufVal::Bool(b[0] != 0))
        }
        8 => Ok(GgufVal::Str(read_gguf_string(r)?)),
        9 => {
            let elem_type = read_u32(r)?;
            let count = read_u64(r)? as usize;
            if elem_type == 8 {
                let mut arr = Vec::with_capacity(count);
                for _ in 0..count {
                    arr.push(read_gguf_string(r)?);
                }
                Ok(GgufVal::StrArray(arr))
            } else {
                for _ in 0..count {
                    let _ = read_gguf_val(r, elem_type)?;
                }
                Ok(GgufVal::Other)
            }
        }
        10 | 11 | 12 => {
            let _ = read_u64(r)?;
            Ok(GgufVal::Other)
        }
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unknown GGUF metadata type: {}", vtype),
        )),
    }
}

fn get_u32(m: &HashMap<String, GgufVal>, k: &str) -> Option<u32> {
    match m.get(k) {
        Some(GgufVal::U32(v)) => Some(*v),
        Some(GgufVal::I32(v)) => Some(*v as u32),
        _ => None,
    }
}

fn get_f32(m: &HashMap<String, GgufVal>, k: &str) -> Option<f32> {
    match m.get(k) {
        Some(GgufVal::F32(v)) => Some(*v),
        _ => None,
    }
}

fn get_bool(m: &HashMap<String, GgufVal>, k: &str) -> Option<bool> {
    match m.get(k) {
        Some(GgufVal::Bool(v)) => Some(*v),
        _ => None,
    }
}

fn get_str<'a>(m: &'a HashMap<String, GgufVal>, k: &str) -> Option<&'a str> {
    match m.get(k) {
        Some(GgufVal::Str(s)) => Some(s.as_str()),
        _ => None,
    }
}
