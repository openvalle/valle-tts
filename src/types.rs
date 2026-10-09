use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
/// Language control for synthesis; automatic mode also permits mixed text.
pub enum Language {
    #[default]
    /// Infer the language from text; mixed Chinese/English is permitted.
    Auto,
    /// Request Chinese pronunciation.
    Chinese,
    /// Request English pronunciation.
    English,
}
impl Language {
    /// Parse case-insensitive `auto`, `zh`/`zh-cn`/`chinese`, or `en`/`en-us`/`english`.
    ///
    /// # Errors
    /// Returns an error for an unsupported language string.
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
/// A cloneable, thread-safe signal to stop synthesis cooperatively.
/// Cancellation is permanent for all clones; create a new token for later requests.
pub struct CancellationToken(Arc<AtomicBool>);
impl CancellationToken {
    /// Signal cancellation to every clone of this token.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }
    /// Whether cancellation has been signalled.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

#[derive(Debug, Clone)]
/// Per-request synthesis controls, including bounded chunking and cancellation.
pub struct SynthesisOptions {
    /// Requested language; defaults to automatic detection.
    pub language: Language,
    /// Sampling seed; defaults to 42. Reproducibility depends on the backend.
    pub seed: i64,
    /// Hard frame budget per text chunk within `1..=4096`, default 512.
    /// 12.5 Qwen frames correspond to one second. Exhaustion without EOS is an error.
    pub max_tokens: u32,
    /// Sampling temperature within `0..=2`; defaults to 0.9.
    pub temperature: f32,
    /// Unicode character limit per text chunk within `1..=500`, default 160.
    /// Splitting at punctuation bounds text/KV/audio growth for long input.
    pub max_chunk_chars: usize,
    /// Shared stop signal; backends determine interruption boundaries.
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
/// Backend identity, capabilities and output audio format.
pub struct ModelInfo {
    /// Unique model ID used for engine dispatch.
    pub id: String,
    /// Model family name.
    pub family: String,
    /// Supported language controls.
    pub languages: Vec<Language>,
    /// Whether reference-voice conditioning is available.
    pub voice_cloning: bool,
    /// Mono output sample rate in hertz.
    pub sample_rate: u32,
}
#[derive(Debug, Clone)]
/// One owned, mono PCM chunk delivered to the output sink.
pub struct AudioChunk {
    /// Owned finite PCM samples; output may exceed `[-1, 1]` before writing.
    pub samples: Vec<f32>,
    /// Mono output sample rate in hertz.
    pub sample_rate: u32,
}
#[derive(Debug, Clone, Serialize)]
/// Totals returned after successful streamed synthesis.
pub struct SynthesisSummary {
    /// ID of the backend that produced this audio.
    pub model: String,
    /// Mono output sample rate in hertz.
    pub sample_rate: u32,
    /// Total emitted mono sample count, not bytes or milliseconds.
    pub samples: u64,
    /// Number of independently synthesized text chunks.
    pub text_chunks: u64,
}
