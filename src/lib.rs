#![doc = include_str!("../README.md")]
#![deny(missing_docs, rustdoc::broken_intra_doc_links)]

mod audio;
#[cfg(feature = "download")]
pub mod cache;
mod engine;
/// Built-in optional model backends.
pub mod models;
mod output;
mod types;
pub use audio::ReferenceVoice;
pub use engine::{TtsEngine, TtsModel};
pub use output::WavOutput;
pub use types::{
    AudioChunk, CancellationToken, Language, ModelInfo, SynthesisOptions, SynthesisSummary,
};
