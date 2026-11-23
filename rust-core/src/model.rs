use crate::config::{ModelConfig, QuantType};
use crate::quant::WeightMatrix;
use crate::tensor::{apply_rope_head, rms_norm, silu, softmax_inplace};
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

pub struct LayerWeights {
    pub attn_norm: Vec<f32>,
    pub wq: WeightMatrix,
    pub wk: WeightMatrix,
    pub wv: WeightMatrix,
    pub wo: WeightMatrix,
    pub ffn_norm: Vec<f32>,
    pub w_gate: WeightMatrix,
    pub w_up: WeightMatrix,
    pub w_down: WeightMatrix,
}

pub struct Transformer {
    pub config: ModelConfig,
    pub token_embedding: WeightMatrix,
    pub layers: Vec<LayerWeights>,
    pub final_norm: Vec<f32>,
    pub lm_head: Option<WeightMatrix>,
}

pub struct RunState {
    pub x: Vec<f32>,
    pub xb: Vec<f32>,
    pub xb2: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub att: Vec<f32>,
    pub hb: Vec<f32>,
    pub hb2: Vec<f32>,
    pub logits: Vec<f32>,
    pub key_cache: Vec<f32>,
    pub value_cache: Vec<f32>,
}

impl RunState {
    pub fn new(cfg: &ModelConfig) -> Self {
        let kv_dim = cfg.kv_dim();
        let cache_size = cfg.n_layers * cfg.max_seq_len * kv_dim;
        Self {
            x: vec![0.0; cfg.dim],
            xb: vec![0.0; cfg.dim],
            xb2: vec![0.0; cfg.dim],
            q: vec![0.0; cfg.dim],
            k: vec![0.0; kv_dim],
            v: vec![0.0; kv_dim],
            att: vec![0.0; cfg.n_heads * cfg.max_seq_len],
            hb: vec![0.0; cfg.hidden_dim],
            hb2: vec![0.0; cfg.hidden_dim],
            logits: vec![0.0; cfg.vocab_size],
            key_cache: vec![0.0; cache_size],
            value_cache: vec![0.0; cache_size],
        }
    }
}

impl Transformer {
    pub fn forward(&self, token: usize, pos: usize, state: &mut RunState, n_threads: usize) {
        let cfg = &self.config;
        let dim = cfg.dim;
        let kv_dim = cfg.kv_dim();
        let head_dim = cfg.head_dim();
        let kv_mul = cfg.n_heads / cfg.n_kv_heads;
        let inv_sqrt_hd = 1.0 / (head_dim as f32).sqrt();

        self.token_embedding.dequantize_row(token, &mut state.x);

        for l in 0..cfg.n_layers {
            let layer = &self.layers[l];

            rms_norm(&mut state.xb, &state.x, &layer.attn_norm, cfg.rms_eps);
            layer.wq.matvec(&state.xb, &mut state.q, n_threads);
            layer.wk.matvec(&state.xb, &mut state.k, n_threads);
            layer.wv.matvec(&state.xb, &mut state.v, n_threads);

            for h in 0..cfg.n_heads {
                let q_head = &mut state.q[h * head_dim..(h + 1) * head_dim];
                apply_rope_head(q_head, pos, head_dim, cfg.rope_theta);
            }
            for h in 0..cfg.n_kv_heads {
                let k_head = &mut state.k[h * head_dim..(h + 1) * head_dim];
                apply_rope_head(k_head, pos, head_dim, cfg.rope_theta);
            }

            let loff = l * cfg.max_seq_len * kv_dim;
            let cache_pos = loff + pos * kv_dim;
            state.key_cache[cache_pos..cache_pos + kv_dim].copy_from_slice(&state.k);
            state.value_cache[cache_pos..cache_pos + kv_dim].copy_from_slice(&state.v);

            state.xb.fill(0.0);
            for h in 0..cfg.n_heads {
                let q_head = &state.q[h * head_dim..(h + 1) * head_dim];
                let att_slice = &mut state.att[h * cfg.max_seq_len..h * cfg.max_seq_len + pos + 1];
                let kv_h = h / kv_mul;

                for t in 0..=pos {
                    let k_off = loff + t * kv_dim + kv_h * head_dim;
                    let k_head = &state.key_cache[k_off..k_off + head_dim];
                    let mut score = 0.0f32;
                    for i in 0..head_dim {
                        score += q_head[i] * k_head[i];
                    }
                    att_slice[t] = score * inv_sqrt_hd;
                }

                softmax_inplace(att_slice);

                let out_head = &mut state.xb[h * head_dim..(h + 1) * head_dim];
                for t in 0..=pos {
                    let a = att_slice[t];
                    let v_off = loff + t * kv_dim + kv_h * head_dim;
                    let v_head = &state.value_cache[v_off..v_off + head_dim];
                    for i in 0..head_dim {
                        out_head[i] += a * v_head[i];
                    }
                }
            }

            layer.wo.matvec(&state.xb, &mut state.xb2, n_threads);
            for i in 0..dim {
                state.x[i] += state.xb2[i];
            }

            rms_norm(&mut state.xb, &state.x, &layer.ffn_norm, cfg.rms_eps);
            layer.w_gate.matvec(&state.xb, &mut state.hb, n_threads);
            layer.w_up.matvec(&state.xb, &mut state.hb2, n_threads);
            for i in 0..cfg.hidden_dim {
                state.hb[i] = silu(state.hb[i]) * state.hb2[i];
            }
            layer.w_down.matvec(&state.hb, &mut state.xb2, n_threads);
            for i in 0..dim {
                state.x[i] += state.xb2[i];
            }
        }

        rms_norm(&mut state.xb, &state.x, &self.final_norm, cfg.rms_eps);
        match &self.lm_head {
            Some(head) => head.matvec(&state.xb, &mut state.logits, n_threads),
            None => self.token_embedding.matvec(&state.xb, &mut state.logits, n_threads),
        }
    }

    pub fn load_oryn<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let mut r = BufReader::new(File::open(path)?);
        let config = ModelConfig::read_from(&mut r)?;
        let q = config.quant_type;
        let dim = config.dim;
        let kv_dim = config.kv_dim();
        let hidden = config.hidden_dim;

        let token_embedding = WeightMatrix::read_from(&mut r, config.vocab_size, dim, q)?;
        let mut layers = Vec::with_capacity(config.n_layers);
        for _ in 0..config.n_layers {
            let attn_norm = read_f32_vec(&mut r, dim)?;
            let wq = WeightMatrix::read_from(&mut r, dim, dim, q)?;
            let wk = WeightMatrix::read_from(&mut r, kv_dim, dim, q)?;
            let wv = WeightMatrix::read_from(&mut r, kv_dim, dim, q)?;
            let wo = WeightMatrix::read_from(&mut r, dim, dim, q)?;
            let ffn_norm = read_f32_vec(&mut r, dim)?;
            let w_gate = WeightMatrix::read_from(&mut r, hidden, dim, q)?;
            let w_up = WeightMatrix::read_from(&mut r, hidden, dim, q)?;
            let w_down = WeightMatrix::read_from(&mut r, dim, hidden, q)?;
            layers.push(LayerWeights {
                attn_norm,
                wq,
                wk,
                wv,
                wo,
                ffn_norm,
                w_gate,
                w_up,
                w_down,
            });
        }
        let final_norm = read_f32_vec(&mut r, dim)?;
        let lm_head = if config.tied_embeddings {
            None
        } else {
            Some(WeightMatrix::read_from(&mut r, config.vocab_size, dim, q)?)
        };

        Ok(Self {
            config,
            token_embedding,
            layers,
            final_norm,
            lm_head,
        })
    }

    pub fn estimated_ram_mb(&self) -> f64 {
        let bytes_per_weight = match self.config.quant_type {
            QuantType::F32 => 4.0,
            QuantType::Q8_0 => 1.125,
            QuantType::Q4_0 => 0.625,
        };
        let weight_bytes = (self.config.param_count() as f64) * bytes_per_weight;
        let kv_bytes = (2 * self.config.n_layers * self.config.max_seq_len * self.config.kv_dim() * 4) as f64;
        (weight_bytes + kv_bytes) / (1024.0 * 1024.0)
    }
}

fn read_f32_vec<R: Read>(r: &mut R, len: usize) -> io::Result<Vec<f32>> {
    let mut raw = vec![0u8; len * 4];
    r.read_exact(&mut raw)?;
    Ok(raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}
