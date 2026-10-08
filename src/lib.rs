//! Local TTS with model-independent voice references and streamed PCM output.
mod audio;
#[cfg(feature = "download")]
pub mod cache;
mod engine;
pub mod models;
mod output;
mod types;
pub use audio::ReferenceVoice;
pub use engine::{TtsEngine, TtsModel};
pub use output::WavOutput;
pub use types::{
    AudioChunk, CancellationToken, Language, ModelInfo, SynthesisOptions, SynthesisSummary,
};
