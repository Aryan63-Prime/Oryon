use std::io::{self, Read};

pub const MAGIC: &[u8; 4] = b"ORYN";
pub const VERSION: u32 = 1;
pub const HEADER_SIZE: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum QuantType {
    F32 = 0,
    Q8_0 = 1,
    Q4_0 = 2,
}

impl QuantType {
    pub fn from_u32(v: u32) -> io::Result<Self> {
        match v {
            0 => Ok(Self::F32),
            1 => Ok(Self::Q8_0),
            2 => Ok(Self::Q4_0),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported quant type id: {}", v),
            )),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::Q8_0 => "Q8_0",
            Self::Q4_0 => "Q4_0",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub quant_type: QuantType,
    pub vocab_size: usize,
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub max_seq_len: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
    pub tied_embeddings: bool,
    pub add_bos: bool,
}

impl ModelConfig {
    pub fn head_dim(&self) -> usize {
        self.dim / self.n_heads
    }

    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim()
    }

    pub fn validate(&self) -> io::Result<()> {
        if self.dim == 0 || self.n_heads == 0 || self.n_kv_heads == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Model dimensions and head counts must be > 0",
            ));
        }
        if self.dim % self.n_heads != 0 || self.n_heads % self.n_kv_heads != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid GQA head configuration",
            ));
        }
        if self.quant_type != QuantType::F32 && (self.dim % 32 != 0 || self.hidden_dim % 32 != 0) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Quantized models require dim and hidden_dim to be multiples of 32",
            ));
        }
        Ok(())
    }

    pub fn read_from<R: Read>(r: &mut R) -> io::Result<Self> {
        let mut buf = [0u8; HEADER_SIZE];
        r.read_exact(&mut buf)?;
        if &buf[0..4] != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid checkpoint magic (expected ORYN)",
            ));
        }
        let version = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        if version != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported checkpoint version: {}", version),
            ));
        }
        let cfg = Self {
            quant_type: QuantType::from_u32(u32::from_le_bytes(buf[8..12].try_into().unwrap()))?,
            vocab_size: u32::from_le_bytes(buf[12..16].try_into().unwrap()) as usize,
            dim: u32::from_le_bytes(buf[16..20].try_into().unwrap()) as usize,
            hidden_dim: u32::from_le_bytes(buf[20..24].try_into().unwrap()) as usize,
            n_layers: u32::from_le_bytes(buf[24..28].try_into().unwrap()) as usize,
            n_heads: u32::from_le_bytes(buf[28..32].try_into().unwrap()) as usize,
            n_kv_heads: u32::from_le_bytes(buf[32..36].try_into().unwrap()) as usize,
            max_seq_len: u32::from_le_bytes(buf[36..40].try_into().unwrap()) as usize,
            rope_theta: f32::from_le_bytes(buf[40..44].try_into().unwrap()),
            rms_eps: f32::from_le_bytes(buf[44..48].try_into().unwrap()),
            tied_embeddings: u32::from_le_bytes(buf[48..52].try_into().unwrap()) != 0,
            add_bos: false,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn param_count(&self) -> usize {
        let emb = self.vocab_size * self.dim;
        let per_layer = self.dim
            + self.dim * self.dim
            + self.kv_dim() * self.dim
            + self.kv_dim() * self.dim
            + self.dim * self.dim
            + self.dim
            + self.hidden_dim * self.dim
            + self.hidden_dim * self.dim
            + self.dim * self.hidden_dim;
        let lm_head = if self.tied_embeddings { 0 } else { emb };
        emb + self.n_layers * per_layer + self.dim + lm_head
    }
}
