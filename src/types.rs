use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Language {
    #[default]
    Auto,
    Chinese,
    English,
}
impl Language {
    pub fn parse(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "zh" | "zh-cn" | "chinese" => Ok(Self::Chinese),
            "en" | "en-us" | "english" => Ok(Self::English),
            _ => anyhow::bail!("unsupported language: {value}; use auto, zh or en"),
        }
    }
    #[cfg(feature = "qwen3")]
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Chinese => "chinese",
            Self::English => "english",
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct CancellationToken(Arc<AtomicBool>);
impl CancellationToken {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

#[derive(Debug, Clone)]
pub struct SynthesisOptions {
    pub language: Language,
    pub seed: i64,
    /// Hard frame budget per text chunk. 12.5 frames correspond to one second.
    pub max_tokens: u32,
    pub temperature: f32,
    /// Bound text/KV/audio growth for long input by splitting at punctuation.
    pub max_chunk_chars: usize,
    pub cancellation: CancellationToken,
}
impl Default for SynthesisOptions {
    fn default() -> Self {
        Self {
            language: Language::Auto,
            seed: 42,
            max_tokens: 512,
            temperature: 0.9,
            max_chunk_chars: 160,
            cancellation: CancellationToken::default(),
        }
    }
}
impl SynthesisOptions {
    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(
            (1..=4096).contains(&self.max_tokens),
            "max_tokens must be 1..=4096"
        );
        ensure!(
            (1..=500).contains(&self.max_chunk_chars),
            "max_chunk_chars must be 1..=500"
        );
        ensure!(
            self.temperature.is_finite() && (0.0..=2.0).contains(&self.temperature),
            "temperature must be finite and within 0..=2"
        );
        ensure!(!self.cancellation.is_cancelled(), "synthesis cancelled");
        Ok(())
    }
}
#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub family: String,
    pub languages: Vec<Language>,
    pub voice_cloning: bool,
    pub sample_rate: u32,
}
#[derive(Debug, Clone)]
pub struct AudioChunk {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}
#[derive(Debug, Clone, Serialize)]
pub struct SynthesisSummary {
    pub model: String,
    pub sample_rate: u32,
    pub samples: u64,
    pub text_chunks: u64,
}
