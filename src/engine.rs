use crate::{AudioChunk, ModelInfo, ReferenceVoice, SynthesisOptions, SynthesisSummary};
use anyhow::{Result, ensure};
use std::collections::BTreeMap;

/// Extension point for model families, sharing reference voice and streamed PCM contracts.
pub trait TtsModel: Send {
    /// Return backend identity and capabilities.
    fn info(&self) -> ModelInfo;
    /// Synthesize text conditioned on a reference voice and emit owned mono PCM.
    /// The sink can run on a model's compute worker; it must be Send.
    /// Backends must propagate sink failures and return an error on failed or
    /// incomplete generation. A successful summary counts all emitted samples.
    fn synthesize(
        &mut self,
        text: &str,
        voice: &ReferenceVoice,
        options: &SynthesisOptions,
        emit: &mut (dyn FnMut(AudioChunk) -> Result<()> + Send),
    ) -> Result<SynthesisSummary>;
}
#[derive(Default)]
/// A registry that dispatches requests to independently registered models.
pub struct TtsEngine {
    models: BTreeMap<String, Box<dyn TtsModel>>,
}
impl TtsEngine {
    /// Create an empty model registry.
    pub fn new() -> Self {
        Self::default()
    }
    /// Register a model under its reported ID.
    ///
    /// # Errors
    /// Returns an error if the ID is empty or already registered.
    pub fn register(&mut self, model: impl TtsModel + 'static) -> Result<()> {
        let id = model.info().id;
        ensure!(!id.is_empty(), "model ID is empty");
        ensure!(
            !self.models.contains_key(&id),
            "model already registered: {id}"
        );
        self.models.insert(id, Box::new(model));
        Ok(())
    }
    /// Return model capabilities in ascending model-ID order.
    pub fn models(&self) -> Vec<ModelInfo> {
        self.models.values().map(|m| m.info()).collect()
    }
    /// Stream synthesis through a registered backend. The sink may execute
    /// on its compute worker and must not assume it runs on the calling thread.
    ///
    /// # Errors
    /// Rejects empty text, invalid options, cancellation and unknown model IDs.
    /// Propagates reference, inference and sink failures from the backend.
    pub fn synthesize(
        &mut self,
        model: &str,
        text: &str,
        voice: &ReferenceVoice,
        options: &SynthesisOptions,
        emit: &mut (dyn FnMut(AudioChunk) -> Result<()> + Send),
    ) -> Result<SynthesisSummary> {
        options.validate()?;
        ensure!(!text.trim().is_empty(), "text is empty");
        let backend = self
            .models
            .get_mut(model)
            .ok_or_else(|| anyhow::anyhow!("model is not registered: {model}"))?;
        backend.synthesize(text, voice, options, emit)
    }
}
