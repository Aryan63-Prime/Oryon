use crate::config::QuantType;
use std::io::{self, Read};
use std::thread;

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
}

impl WeightMatrix {
    pub fn rows(&self) -> usize {
        match self {
            Self::F32 { rows, .. } | Self::Q8_0 { rows, .. } | Self::Q4_0 { rows, .. } => *rows,
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Self::F32 { cols, .. } | Self::Q8_0 { cols, .. } | Self::Q4_0 { cols, .. } => *cols,
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
        }
    }
}

#[inline]
fn quantize_vec_q8(x: &[f32]) -> (Vec<f32>, Vec<i8>) {
    let n_blocks = x.len() / BLOCK_SIZE;
    let mut scales = Vec::with_capacity(n_blocks);
    let mut quants = Vec::with_capacity(x.len());
    for block in x.chunks_exact(BLOCK_SIZE) {
        let mut max_abs = 0.0f32;
        for &v in block {
            let a = v.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
        let inv = 1.0 / scale;
        scales.push(scale);
        for &v in block {
            quants.push((v * inv).round().clamp(-127.0, 127.0) as i8);
        }
    }
    (scales, quants)
}

#[inline]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let mut i = 0;
    let n = a.len();
    while i + 4 <= n {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
        i += 4;
    }
    let mut sum = (s0 + s1) + (s2 + s3);
    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

#[inline]
fn dot_q8_row(w_scales: &[f32], w_quants: &[i8], x_scales: &[f32], x_quants: &[i8]) -> f32 {
    let mut acc = 0.0f32;
    for b in 0..w_scales.len() {
        let ws = &w_quants[b * BLOCK_SIZE..(b + 1) * BLOCK_SIZE];
        let xs = &x_quants[b * BLOCK_SIZE..(b + 1) * BLOCK_SIZE];
        acc += (dot_i8_32(ws, xs) as f32) * (w_scales[b] * x_scales[b]);
    }
    acc
}

#[inline]
fn dot_q4_row(w_scales: &[f32], w_packed: &[u8], x_scales: &[f32], x_quants: &[i8]) -> f32 {
    let mut acc = 0.0f32;
    for b in 0..w_scales.len() {
        let wp = &w_packed[b * (BLOCK_SIZE / 2)..(b + 1) * (BLOCK_SIZE / 2)];
        let xs = &x_quants[b * BLOCK_SIZE..(b + 1) * BLOCK_SIZE];
        let mut sum: i32 = 0;
        for i in 0..(BLOCK_SIZE / 2) {
            let byte = wp[i];
            let q0 = (byte & 0x0F) as i32 - 8;
            let q1 = ((byte >> 4) & 0x0F) as i32 - 8;
            sum += q0 * (xs[2 * i] as i32) + q1 * (xs[2 * i + 1] as i32);
        }
        acc += (sum as f32) * (w_scales[b] * x_scales[b]);
    }
    acc
}

#[inline]
fn dot_i8_32(a: &[i8], b: &[i8]) -> i32 {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            use std::arch::aarch64::*;
            let a0 = vld1q_s8(a.as_ptr());
            let b0 = vld1q_s8(b.as_ptr());
            let a1 = vld1q_s8(a.as_ptr().add(16));
            let b1 = vld1q_s8(b.as_ptr().add(16));

            let p0_lo = vmull_s8(vget_low_s8(a0), vget_low_s8(b0));
            let p0_hi = vmull_s8(vget_high_s8(a0), vget_high_s8(b0));
            let p1_lo = vmull_s8(vget_low_s8(a1), vget_low_s8(b1));
            let p1_hi = vmull_s8(vget_high_s8(a1), vget_high_s8(b1));

            let sum0 = vaddq_s32(vpaddlq_s16(p0_lo), vpaddlq_s16(p0_hi));
            let sum1 = vaddq_s32(vpaddlq_s16(p1_lo), vpaddlq_s16(p1_hi));
            return vaddvq_s32(vaddq_s32(sum0, sum1));
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut sum: i32 = 0;
        for i in 0..BLOCK_SIZE {
            sum += (a[i] as i32) * (b[i] as i32);
        }
        sum
    }
}
