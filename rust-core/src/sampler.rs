#[derive(Clone)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub repeat_window: usize,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            min_p: 0.05,
            repeat_penalty: 1.1,
            repeat_window: 64,
        }
    }
}

pub struct Sampler {
    pub cfg: SamplerConfig,
    rng_state: u64,
}

impl Sampler {
    pub fn new(cfg: SamplerConfig, seed: u64) -> Self {
        Self {
            cfg,
            rng_state: if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed },
        }
    }

    #[inline]
    fn next_f32(&mut self) -> f32 {
        let mut x = self.rng_state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng_state = x;
        let val = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((val >> 40) as f32) / ((1u64 << 24) as f32)
    }

    pub fn sample(&mut self, logits: &mut [f32], history: &[u32]) -> u32 {
        if self.cfg.repeat_penalty > 1.0 && !history.is_empty() {
            let start = history.len().saturating_sub(self.cfg.repeat_window);
            for &tok in &history[start..] {
                let idx = tok as usize;
                if idx < logits.len() {
                    if logits[idx] > 0.0 {
                        logits[idx] /= self.cfg.repeat_penalty;
                    } else {
                        logits[idx] *= self.cfg.repeat_penalty;
                    }
                }
            }
        }

        if self.cfg.temperature <= 1e-6 {
            let mut best_idx = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for (i, &v) in logits.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best_idx = i;
                }
            }
            return best_idx as u32;
        }

        let inv_temp = 1.0 / self.cfg.temperature;
        let mut max_l = f32::NEG_INFINITY;
        for v in logits.iter_mut() {
            *v *= inv_temp;
            if *v > max_l {
                max_l = *v;
            }
        }

        let mut probs: Vec<(usize, f32)> = Vec::with_capacity(logits.len());
        let mut sum = 0.0f32;
        for (i, &v) in logits.iter().enumerate() {
            let p = (v - max_l).exp();
            sum += p;
            probs.push((i, p));
        }
        let inv_sum = 1.0 / sum;
        for pair in probs.iter_mut() {
            pair.1 *= inv_sum;
        }

        probs.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        let max_p = probs[0].1;
        if self.cfg.min_p > 0.0 {
            let thresh = max_p * self.cfg.min_p;
            let keep = probs.iter().take_while(|&&(_, p)| p >= thresh).count().max(1);
            probs.truncate(keep);
        }

        if self.cfg.top_k > 0 && self.cfg.top_k < probs.len() {
            probs.truncate(self.cfg.top_k);
        }

        if self.cfg.top_p > 0.0 && self.cfg.top_p < 1.0 {
            let mut cum = 0.0f32;
            let mut cut = probs.len();
            for (i, &(_, p)) in probs.iter().enumerate() {
                cum += p;
                if cum >= self.cfg.top_p {
                    cut = i + 1;
                    break;
                }
            }
            probs.truncate(cut.max(1));
        }

        let total: f32 = probs.iter().map(|&(_, p)| p).sum();
        let r = self.next_f32() * total;
        let mut acc = 0.0f32;
        for &(idx, p) in &probs {
            acc += p;
            if acc >= r {
                return idx as u32;
            }
        }
        probs[0].0 as u32
    }
}
