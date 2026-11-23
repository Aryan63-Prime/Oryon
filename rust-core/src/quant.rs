use crate::config::QuantType;
use std::io::{self, Read};
use std::thread;
use std::sync::Arc;
use memmap2::Mmap;

pub const BLOCK_SIZE: usize = 32;

/// Raised thread threshold (256 rows) avoids OS thread-spawn overhead on small matrices
/// while still saturating all 6 big Oryon V2 cores on 1B–7B models.
const PARALLEL_ROW_THRESHOLD: usize = 256;

#[derive(Clone)]
pub enum WeightMatrix {
    F32 {
        rows: usize,
        cols: usize,
        data: Vec<f32>,
    },
    Q8_0 {
        rows: usize,
        cols: usize,
        scales: Vec<f32>,
        quants: Vec<i8>,
    },
    Q4_0 {
        rows: usize,
        cols: usize,
        scales: Vec<f32>,
        packed: Vec<u8>,
    },
    Q4_K {
        rows: usize,
        cols: usize,
        blocks: Vec<u8>,
    },
    Q6_K {
        rows: usize,
        cols: usize,
        blocks: Vec<u8>,
    },
    F32Mmap {
        rows: usize,
        cols: usize,
        mmap: Arc<Mmap>,
        offset: usize,
    },
    Q4KMmap {
        rows: usize,
        cols: usize,
        mmap: Arc<Mmap>,
        offset: usize,
    },
    Q6KMmap {
        rows: usize,
        cols: usize,
        mmap: Arc<Mmap>,
        offset: usize,
    },
}

impl WeightMatrix {
    pub fn rows(&self) -> usize {
        match self {
            Self::F32 { rows, .. } | Self::Q8_0 { rows, .. } | Self::Q4_0 { rows, .. }
            | Self::Q4_K { rows, .. } | Self::Q6_K { rows, .. }
            | Self::F32Mmap { rows, .. } | Self::Q4KMmap { rows, .. } | Self::Q6KMmap { rows, .. } => *rows,
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Self::F32 { cols, .. } | Self::Q8_0 { cols, .. } | Self::Q4_0 { cols, .. }
            | Self::Q4_K { cols, .. } | Self::Q6_K { cols, .. }
            | Self::F32Mmap { cols, .. } | Self::Q4KMmap { cols, .. } | Self::Q6KMmap { cols, .. } => *cols,
        }
    }

    pub fn dequantize_row(&self, row: usize, out: &mut [f32]) {
        let cols = self.cols();
        match self {
            Self::F32 { data, .. } => {
                let start = row * cols;
                out.copy_from_slice(&data[start..start + cols]);
            }
            Self::Q8_0 { scales, quants, .. } => {
                let blocks_per_row = cols / BLOCK_SIZE;
                let b0 = row * blocks_per_row;
                for b in 0..blocks_per_row {
                    let scale = scales[b0 + b];
                    let q_slice = &quants[(b0 + b) * BLOCK_SIZE..(b0 + b + 1) * BLOCK_SIZE];
                    let dst = &mut out[b * BLOCK_SIZE..(b + 1) * BLOCK_SIZE];
                    for i in 0..BLOCK_SIZE {
                        dst[i] = (q_slice[i] as f32) * scale;
                    }
                }
            }
            Self::Q4_0 { scales, packed, .. } => {
                let blocks_per_row = cols / BLOCK_SIZE;
                let b0 = row * blocks_per_row;
                for b in 0..blocks_per_row {
                    let scale = scales[b0 + b];
                    let p_slice = &packed[(b0 + b) * (BLOCK_SIZE / 2)..(b0 + b + 1) * (BLOCK_SIZE / 2)];
                    let dst = &mut out[b * BLOCK_SIZE..(b + 1) * BLOCK_SIZE];
                    for i in 0..(BLOCK_SIZE / 2) {
                        let byte = p_slice[i];
                        dst[2 * i] = ((byte & 0x0F) as i32 - 8) as f32 * scale;
                        dst[2 * i + 1] = (((byte >> 4) & 0x0F) as i32 - 8) as f32 * scale;
                    }
                }
            }
            Self::Q4_K { blocks, .. } => dequant_q4_k_row(cols, row, blocks, out),
            Self::Q6_K { blocks, .. } => dequant_q6_k_row(cols, row, blocks, out),
            Self::F32Mmap { mmap, offset, .. } => {
                let start = offset + row * cols * 4;
                let bytes = &mmap[start..start + cols * 4];
                for i in 0..cols {
                    out[i] = f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
                }
            }
            Self::Q4KMmap { mmap, offset, .. } => {
                let super_blocks = cols / 256;
                let start = offset + row * super_blocks * 144;
                dequant_q4_k_row(cols, 0, &mmap[start..start + super_blocks * 144], out);
            }
            Self::Q6KMmap { mmap, offset, .. } => {
                let super_blocks = cols / 256;
                let start = offset + row * super_blocks * 210;
                dequant_q6_k_row(cols, 0, &mmap[start..start + super_blocks * 210], out);
            }
        }
    }

    pub fn matvec(&self, x: &[f32], out: &mut [f32], n_threads: usize) {
        let rows = self.rows();
        let cols = self.cols();
        let use_threads = if rows < PARALLEL_ROW_THRESHOLD {
            1
        } else {
            n_threads.max(1).min(rows)
        };

        match self {
            Self::F32 { data, .. } => {
                if use_threads == 1 {
                    for r in 0..rows {
                        out[r] = dot_f32(&data[r * cols..(r + 1) * cols], x);
                    }
                } else {
                    let chunk = (rows + use_threads - 1) / use_threads;
                    thread::scope(|s| {
                        for (t, out_chunk) in out.chunks_mut(chunk).enumerate() {
                            let r_start = t * chunk;
                            s.spawn(move || {
                                for (i, o) in out_chunk.iter_mut().enumerate() {
                                    let r = r_start + i;
                                    *o = dot_f32(&data[r * cols..(r + 1) * cols], x);
                                }
                            });
                        }
                    });
                }
            }
            Self::Q4_K { blocks, .. } => {
                for r in 0..rows {
                    out[r] = dot_q4_k_row(blocks, r, cols, x);
                }
            }
            Self::Q6_K { blocks, .. } => {
                for r in 0..rows {
                    out[r] = dot_q6_k_row(blocks, r, cols, x);
                }
            }
            Self::F32Mmap { mmap, offset, .. } => {
                for r in 0..rows {
                    let start = offset + r * cols * 4;
                    let mut sum = 0.0f32;
                    for i in 0..cols {
                        let w = f32::from_le_bytes(mmap[start + i * 4..start + i * 4 + 4].try_into().unwrap());
                        sum += w * x[i];
                    }
                    out[r] = sum;
                }
            }
            Self::Q4KMmap { mmap, offset, .. } => {
                for r in 0..rows {
                    let super_blocks = cols / 256;
                    let start = offset + r * super_blocks * 144;
                    let mut row = vec![0.0f32; cols];
                    dequant_q4_k_row(cols, 0, &mmap[start..start + super_blocks * 144], &mut row);
                    out[r] = row.iter().zip(x).map(|(&w, &v)| w * v).sum();
                }
            }
            Self::Q6KMmap { mmap, offset, .. } => {
                for r in 0..rows {
                    let super_blocks = cols / 256;
                    let start = offset + r * super_blocks * 210;
                    let mut row = vec![0.0f32; cols];
                    dequant_q6_k_row(cols, 0, &mmap[start..start + super_blocks * 210], &mut row);
                    out[r] = row.iter().zip(x).map(|(&w, &v)| w * v).sum();
                }
            }
            Self::Q8_0 { scales, quants, .. } => {
                let (x_scales, x_quants) = quantize_vec_q8(x);
                let blocks_per_row = cols / BLOCK_SIZE;
                if use_threads == 1 {
                    for r in 0..rows {
                        out[r] = dot_q8_row(
                            &scales[r * blocks_per_row..(r + 1) * blocks_per_row],
                            &quants[r * cols..(r + 1) * cols],
                            &x_scales,
                            &x_quants,
                        );
                    }
                } else {
                    let chunk = (rows + use_threads - 1) / use_threads;
                    let xs_ref = &x_scales;
                    let xq_ref = &x_quants;
                    thread::scope(|s| {
                        for (t, out_chunk) in out.chunks_mut(chunk).enumerate() {
                            let r_start = t * chunk;
                            s.spawn(move || {
                                for (i, o) in out_chunk.iter_mut().enumerate() {
                                    let r = r_start + i;
                                    *o = dot_q8_row(
                                        &scales[r * blocks_per_row..(r + 1) * blocks_per_row],
                                        &quants[r * cols..(r + 1) * cols],
                                        xs_ref,
                                        xq_ref,
                                    );
                                }
                            });
                        }
                    });
                }
            }
            Self::Q4_0 { scales, packed, .. } => {
                let (x_scales, x_quants) = quantize_vec_q8(x);
                let blocks_per_row = cols / BLOCK_SIZE;
                let bytes_per_row = cols / 2;
                if use_threads == 1 {
                    for r in 0..rows {
                        out[r] = dot_q4_row(
                            &scales[r * blocks_per_row..(r + 1) * blocks_per_row],
                            &packed[r * bytes_per_row..(r + 1) * bytes_per_row],
                            &x_scales,
                            &x_quants,
                        );
                    }
                } else {
                    let chunk = (rows + use_threads - 1) / use_threads;
                    let xs_ref = &x_scales;
                    let xq_ref = &x_quants;
                    thread::scope(|s| {
                        for (t, out_chunk) in out.chunks_mut(chunk).enumerate() {
                            let r_start = t * chunk;
                            s.spawn(move || {
                                for (i, o) in out_chunk.iter_mut().enumerate() {
                                    let r = r_start + i;
                                    *o = dot_q4_row(
                                        &scales[r * blocks_per_row..(r + 1) * blocks_per_row],
                                        &packed[r * bytes_per_row..(r + 1) * bytes_per_row],
                                        xs_ref,
                                        xq_ref,
                                    );
                                }
                            });
                        }
                    });
                }
            }
        }
    }

    pub fn read_from<R: Read>(
        r: &mut R,
        rows: usize,
        cols: usize,
        qtype: QuantType,
    ) -> io::Result<Self> {
        match qtype {
            QuantType::F32 => {
                let mut raw = vec![0u8; rows * cols * 4];
                r.read_exact(&mut raw)?;
                let data = raw
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                Ok(Self::F32 { rows, cols, data })
            }
            QuantType::Q8_0 => {
                let n_blocks = (rows * cols) / BLOCK_SIZE;
                let mut scales = Vec::with_capacity(n_blocks);
                let mut quants = Vec::with_capacity(rows * cols);
                let mut block_buf = [0u8; 4 + BLOCK_SIZE];
                for _ in 0..n_blocks {
                    r.read_exact(&mut block_buf)?;
                    scales.push(f32::from_le_bytes(block_buf[0..4].try_into().unwrap()));
                    for &b in &block_buf[4..] {
                        quants.push(b as i8);
                    }
                }
                Ok(Self::Q8_0 {
                    rows,
                    cols,
                    scales,
                    quants,
                })
            }
            QuantType::Q4_0 => {
                let n_blocks = (rows * cols) / BLOCK_SIZE;
                let mut scales = Vec::with_capacity(n_blocks);
                let mut packed = Vec::with_capacity((rows * cols) / 2);
                let mut block_buf = [0u8; 4 + BLOCK_SIZE / 2];
                for _ in 0..n_blocks {
                    r.read_exact(&mut block_buf)?;
                    scales.push(f32::from_le_bytes(block_buf[0..4].try_into().unwrap()));
                    packed.extend_from_slice(&block_buf[4..]);
                }
                Ok(Self::Q4_0 {
                    rows,
                    cols,
                    scales,
                    packed,
                })
            }
            QuantType::Q4_K | QuantType::Q6_K => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "K-quant ORYN checkpoints are not supported",
            )),
        }
    }
}

#[inline]
fn dot_i8_scalar(a: &[i8], b: &[i8]) -> i32 {
    let mut sum: i32 = 0;
    for i in 0..BLOCK_SIZE {
        sum += (a[i] as i32) * (b[i] as i32);
    }
    sum
}

fn quantize_vec_q8(x: &[f32]) -> (Vec<f32>, Vec<i8>) {
    let n_blocks = x.len().div_ceil(BLOCK_SIZE);
    let mut scales = Vec::with_capacity(n_blocks);
    let mut quants = Vec::with_capacity(x.len());
    for block in x.chunks(BLOCK_SIZE) {
        let mut max_abs = 0.0f32;
        for &v in block { max_abs = max_abs.max(v.abs()); }
        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
        scales.push(scale);
        for &v in block {
            quants.push((v / scale).round().clamp(-127.0, 127.0) as i8);
        }
    }
    (scales, quants)
}

fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

fn dot_q8_row(w_scales: &[f32], w_quants: &[i8], x_scales: &[f32], x_quants: &[i8]) -> f32 {
    let mut acc = 0.0f32;
    for b in 0..w_scales.len() {
        let ws = &w_quants[b * BLOCK_SIZE..(b + 1) * BLOCK_SIZE];
        let xs = &x_quants[b * BLOCK_SIZE..(b + 1) * BLOCK_SIZE];
        acc += dot_i8_scalar(ws, xs) as f32 * w_scales[b] * x_scales[b];
    }
    acc
}

fn dot_q4_row(w_scales: &[f32], w_packed: &[u8], x_scales: &[f32], x_quants: &[i8]) -> f32 {
    let mut acc = 0.0f32;
    for b in 0..w_scales.len() {
        let wp = &w_packed[b * 16..(b + 1) * 16];
        let xs = &x_quants[b * BLOCK_SIZE..(b + 1) * BLOCK_SIZE];
        let mut sum = 0i32;
        for i in 0..16 {
            let q0 = (wp[i] & 15) as i32 - 8;
            let q1 = (wp[i] >> 4) as i32 - 8;
            sum += q0 * xs[2 * i] as i32 + q1 * xs[2 * i + 1] as i32;
        }
        acc += sum as f32 * w_scales[b] * x_scales[b];
    }
    acc
}

const QK_K: usize = 256;
const BLOCK_Q4_K: usize = 144;
const BLOCK_Q6_K: usize = 210;

fn f16_le(bytes: &[u8]) -> f32 {
    let h = u16::from_le_bytes([bytes[0], bytes[1]]);
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 31) as u32;
    let frac = (h & 1023) as u32;
    f32::from_bits(if exp == 0 {
        sign
    } else if exp == 31 {
        sign | (255 << 23) | (frac << 13)
    } else {
        sign | ((exp + 112) << 23) | (frac << 13)
    })
}

fn dequant_q4_k_row(cols: usize, row: usize, blocks: &[u8], out: &mut [f32]) {
    let super_blocks = cols / QK_K;
    let row_offset = row * super_blocks * BLOCK_Q4_K;
    for sb in 0..super_blocks {
        let block = &blocks[row_offset + sb * BLOCK_Q4_K..row_offset + (sb + 1) * BLOCK_Q4_K];
        let d = f16_le(&block[0..2]);
        let dmin = f16_le(&block[2..4]);
        let scales = &block[4..16];
        let qs = &block[16..144];
        for j in 0..8 {
            let q = j * 32;
            let sc = if j < 4 {
                scales[j] & 63
            } else {
                (scales[j + 4] & 15) | ((scales[j - 4] & 0xc0) >> 2)
            };
            let m = if j < 4 {
                scales[j + 4] & 63
            } else {
                (scales[j + 4] >> 4) | ((scales[j - 4] & 0x30) >> 0)
            };
            let dl = d * sc as f32;
            let ml = dmin * m as f32;
            for i in 0..32 {
                let byte = qs[q / 2 + i / 2];
                let qn = if i & 1 == 0 { byte & 15 } else { byte >> 4 };
                out[sb * QK_K + q + i] = dl * qn as f32 - ml;
            }
        }
    }
}

fn dequant_q6_k_row(cols: usize, row: usize, blocks: &[u8], out: &mut [f32]) {
    let super_blocks = cols / QK_K;
    let row_offset = row * super_blocks * BLOCK_Q6_K;
    for sb in 0..super_blocks {
        let block = &blocks[row_offset + sb * BLOCK_Q6_K..row_offset + (sb + 1) * BLOCK_Q6_K];
        let ql = &block[0..128];
        let qh = &block[128..192];
        let sc = &block[192..208];
        let d = f16_le(&block[208..210]);
        for i in 0..128 {
            let scale = sc[i / 16] as i8;
            let high_low = (qh[i / 2] & 3) as i32;
            let high_high = ((qh[i / 2] >> 2) & 3) as i32;
            let qn_low = ((ql[i] & 15) as i32 | (high_low << 4)) - 32;
            let qn_high = ((ql[i] >> 4) as i32 | (high_high << 4)) - 32;
            out[sb * QK_K + i] = d * scale as f32 * qn_low as f32;
            out[sb * QK_K + i + 128] = d * scale as f32 * qn_high as f32;
        }
    }
}

fn dot_q4_k_row(blocks: &[u8], row: usize, cols: usize, x: &[f32]) -> f32 {
    let mut tmp = vec![0.0f32; cols];
    dequant_q4_k_row(cols, row, blocks, &mut tmp);
    tmp.iter().zip(x).map(|(&w, &v)| w * v).sum()
}

fn dot_q6_k_row(blocks: &[u8], row: usize, cols: usize, x: &[f32]) -> f32 {
    let mut tmp = vec![0.0f32; cols];
    dequant_q6_k_row(cols, row, blocks, &mut tmp);
    tmp.iter().zip(x).map(|(&w, &v)| w * v).sum()
}

#[cfg(test)]
mod k_quant_tests {
    use super::*;

    fn f32_to_f16(v: f32) -> u16 {
        ((v.to_bits() >> 16) & 0xffff) as u16
    }

    #[test]
    fn q4_k_dequant_matvec_matches_f32_reference() {
        let mut block = vec![0u8; BLOCK_Q4_K];
        block[0..2].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
        block[2..4].copy_from_slice(&f32_to_f16(0.5).to_le_bytes());
        block[4..12].fill(2);
        for (i, b) in block[16..144].iter_mut().enumerate() {
            *b = if i & 1 == 0 { 0x21 } else { 0x43 };
        }
        let matrix = WeightMatrix::Q4_K { rows: 1, cols: 256, blocks: block };
        let x = vec![1.0f32; 256];
        let mut deq = vec![0.0f32; 256];
        let mut out = vec![0.0f32; 1];
        matrix.dequantize_row(0, &mut deq);
        matrix.matvec(&x, &mut out, 1);
        let reference: f32 = deq.iter().zip(&x).map(|(&w, &v)| w * v).sum();
        assert!((out[0] - reference).abs() < 0.2);
    }

    #[test]
    fn q6_k_dequant_matvec_matches_f32_reference() {
        let mut block = vec![0u8; BLOCK_Q6_K];
        block[208..210].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
        block[192..208].fill(1);
        block[0..128].fill(32);
        block[128..192].fill(0);
        let matrix = WeightMatrix::Q6_K { rows: 1, cols: 256, blocks: block };
        let x = vec![0.5f32; 256];
        let mut deq = vec![0.0f32; 256];
        let mut out = vec![0.0f32; 1];
        matrix.dequantize_row(0, &mut deq);
        matrix.matvec(&x, &mut out, 1);
        let reference: f32 = deq.iter().zip(&x).map(|(&w, &v)| w * v).sum();
        assert!((out[0] - reference).abs() < 0.1);
    }
}
