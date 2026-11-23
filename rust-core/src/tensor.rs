#[inline]
pub fn rms_norm(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    let n = x.len();
    let mut ss = 0.0f32;
    for &v in x {
        ss += v * v;
    }
    let inv_rms = 1.0 / ((ss / n as f32) + eps).sqrt();
    for i in 0..n {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

#[inline]
pub fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let mut max_val = f32::NEG_INFINITY;
    for &v in x.iter() {
        if v > max_val {
            max_val = v;
        }
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
pub fn apply_rope_head(vec: &mut [f32], pos: usize, head_dim: usize, rope_theta: f32) {
    let half = head_dim / 2;
    let pos_f = pos as f32;
    for i in 0..half {
        let freq = 1.0 / rope_theta.powf((2 * i) as f32 / head_dim as f32);
        let angle = pos_f * freq;
        let (sin_a, cos_a) = angle.sin_cos();
        let x0 = vec[i];
        let x1 = vec[i + half];
        vec[i] = x0 * cos_a - x1 * sin_a;
        vec[i + half] = x0 * sin_a + x1 * cos_a;
    }
}
