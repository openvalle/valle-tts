use crate::{AudioChunk, ModelInfo, ReferenceVoice, SynthesisOptions, SynthesisSummary};
use anyhow::{Result, ensure};
use std::collections::BTreeMap;

pub trait TtsModel: Send {
    fn info(&self) -> ModelInfo;
    /// The sink can run on a model's compute worker; it must be Send.
    fn synthesize(
        &mut self,
        text: &str,
        voice: &ReferenceVoice,
        options: &SynthesisOptions,
        emit: &mut (dyn FnMut(AudioChunk) -> Result<()> + Send),
    ) -> Result<SynthesisSummary>;
}
#[derive(Default)]
pub struct TtsEngine {
    models: BTreeMap<String, Box<dyn TtsModel>>,
}
impl TtsEngine {
    pub fn new() -> Self {
        Self::default()
    }
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
    pub fn models(&self) -> Vec<ModelInfo> {
        self.models.values().map(|m| m.info()).collect()
    }
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
