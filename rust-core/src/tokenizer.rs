use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

pub const TOK_MAGIC: &[u8; 4] = b"OTOK";

#[derive(Clone)]
pub struct BpeTokenizer {
    pub vocab: Vec<Vec<u8>>,
    pub merges: Vec<(u32, u32, u32)>,
    pub bos_id: u32,
    pub eos_id: u32,
    /// Optional exact-token lookup map used when loading GGUF vocabularies directly.
    token_to_id: HashMap<Vec<u8>, u32>,
}

impl BpeTokenizer {
    pub fn from_gguf_vocab(tokens: Vec<Vec<u8>>, bos_id: u32, eos_id: u32) -> Self {
        let mut token_to_id = HashMap::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            token_to_id.insert(t.clone(), i as u32);
        }
        Self {
            vocab: tokens,
            merges: Vec::new(),
            bos_id,
            eos_id,
            token_to_id,
        }
    }

    pub fn load<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let mut r = BufReader::new(File::open(path)?);
        let mut header = [0u8; 20];
        r.read_exact(&mut header)?;
        if &header[0..4] != TOK_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid tokenizer magic (expected OTOK)",
            ));
        }
        let vocab_size = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
        let bos_id = u32::from_le_bytes(header[8..12].try_into().unwrap());
        let eos_id = u32::from_le_bytes(header[12..16].try_into().unwrap());
        let num_merges = u32::from_le_bytes(header[16..20].try_into().unwrap()) as usize;

        let mut vocab = Vec::with_capacity(vocab_size);
        let mut len_buf = [0u8; 4];
        for _ in 0..vocab_size {
            r.read_exact(&mut len_buf)?;
            let len = u32::from_le_bytes(len_buf) as usize;
            let mut bytes = vec![0u8; len];
            r.read_exact(&mut bytes)?;
            vocab.push(bytes);
        }

        let mut merges = Vec::with_capacity(num_merges);
        let mut m_buf = [0u8; 12];
        for _ in 0..num_merges {
            r.read_exact(&mut m_buf)?;
            let a = u32::from_le_bytes(m_buf[0..4].try_into().unwrap());
            let b = u32::from_le_bytes(m_buf[4..8].try_into().unwrap());
            let new_id = u32::from_le_bytes(m_buf[8..12].try_into().unwrap());
            merges.push((a, b, new_id));
        }

        Ok(Self {
            vocab,
            merges,
            bos_id,
            eos_id,
            token_to_id: HashMap::new(),
        })
    }

    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<u32> {
        let mut ids = if !self.merges.is_empty() {
            let mut cur: Vec<u32> = text.as_bytes().iter().map(|&b| b as u32).collect();
            for &(a, b, new_id) in &self.merges {
                if cur.len() < 2 {
                    break;
                }
                let mut i = 0;
                let mut next = Vec::with_capacity(cur.len());
                let mut changed = false;
                while i < cur.len() {
                    if i + 1 < cur.len() && cur[i] == a && cur[i + 1] == b {
                        next.push(new_id);
                        i += 2;
                        changed = true;
                    } else {
                        next.push(cur[i]);
                        i += 1;
                    }
                }
                if changed {
                    cur = next;
                }
            }
            cur
        } else {
            // Longest-match greedy tokenizer for GGUF-embedded vocabularies (handles SentencePiece '▁' and GPT-2 'Ġ').
            let normalized = text.replace(' ', "\u{2581}");
            let alt_normalized = text.replace(' ', "\u{0120}");
            let src = if self.token_to_id.contains_key("\u{2581}".as_bytes()) {
                normalized.as_bytes()
            } else if self.token_to_id.contains_key("\u{0120}".as_bytes()) {
                alt_normalized.as_bytes()
            } else {
                text.as_bytes()
            };
            let mut out = Vec::new();
            let mut i = 0;
            while i < src.len() {
                let max_len = (src.len() - i).min(32);
                let mut matched = None;
                for len in (1..=max_len).rev() {
                    if let Some(&id) = self.token_to_id.get(&src[i..i + len]) {
                        matched = Some((id, len));
                        break;
                    }
                }
                if let Some((id, len)) = matched {
                    out.push(id);
                    i += len;
                } else {
                    out.push(src[i] as u32);
                    i += 1;
                }
            }
            out
        };

        if add_bos {
            let mut with_bos = Vec::with_capacity(ids.len() + 1);
            with_bos.push(self.bos_id);
            with_bos.append(&mut ids);
            with_bos
        } else {
            ids
        }
    }

    pub fn decode_token(&self, id: u32) -> String {
        if id == self.bos_id || id == self.eos_id {
            return String::new();
        }
        match self.vocab.get(id as usize) {
            Some(bytes) => String::from_utf8_lossy(bytes)
                .replace('\u{2581}', " ")
                .replace('\u{0120}', " "),
            None => String::new(),
        }
    }
}
