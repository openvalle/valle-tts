use anyhow::Result;
use valle_tts::{
    AudioChunk, CancellationToken, Language, ModelInfo, ReferenceVoice, SynthesisOptions,
    SynthesisSummary, TtsEngine, TtsModel, WavOutput,
};

struct TestModel;
impl TtsModel for TestModel {
    fn info(&self) -> ModelInfo {
        ModelInfo {
            id: "fixture".into(),
            family: "test".into(),
            languages: vec![Language::Chinese, Language::English],
            voice_cloning: true,
            sample_rate: 24000,
        }
    }
    fn synthesize(
        &mut self,
        _: &str,
        _: &ReferenceVoice,
        _: &SynthesisOptions,
        emit: &mut (dyn FnMut(AudioChunk) -> Result<()> + Send),
    ) -> Result<SynthesisSummary> {
        emit(AudioChunk {
            samples: vec![0.25; 100],
            sample_rate: 24000,
        })?;
        Ok(SynthesisSummary {
            model: "fixture".into(),
            sample_rate: 24000,
            samples: 100,
            text_chunks: 1,
        })
    }
}
#[test]
fn independent_model_registration_and_sink_errors() -> Result<()> {
    let mut engine = TtsEngine::new();
    engine.register(TestModel)?;
    assert!(engine.register(TestModel).is_err());
    let voice = ReferenceVoice::new(vec![0.1; 12000], 24000, None)?;
    let options = SynthesisOptions::default();
    assert!(
        engine
            .synthesize("missing", "你好", &voice, &options, &mut |_| Ok(()))
            .is_err()
    );
    let error = engine
        .synthesize("fixture", "Hello", &voice, &options, &mut |_| {
            anyhow::bail!("sink failed")
        })
        .unwrap_err();
    assert!(error.to_string().contains("sink failed"));
    let token = CancellationToken::default();
    token.cancel();
    assert!(
        engine
            .synthesize(
                "fixture",
                "Hello",
                &voice,
                &SynthesisOptions {
                    cancellation: token,
                    ..options
                },
                &mut |_| Ok(())
            )
            .is_err()
    );
    Ok(())
}
#[test]
fn failed_wav_never_replaces_existing_output() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("中文 output.wav");
    std::fs::write(&path, b"original")?;
    {
        let mut writer = WavOutput::new(&path, 24000)?;
        writer.write(AudioChunk {
            samples: vec![0.2; 50],
            sample_rate: 24000,
        })?;
        assert!(
            writer
                .write(AudioChunk {
                    samples: vec![f32::NAN],
                    sample_rate: 24000
                })
                .is_err()
        );
    }
    assert_eq!(std::fs::read(&path)?, b"original");
    let mut writer = WavOutput::new(&path, 24000)?;
    writer.write(AudioChunk {
        samples: vec![0.25; 100],
        sample_rate: 24000,
    })?;
    writer.finish()?;
    let reader = hound::WavReader::open(&path)?;
    assert_eq!(reader.duration(), 100);
    assert_eq!(reader.spec().sample_rate, 24000);
    assert_eq!(reader.spec().bits_per_sample, 16);
    Ok(())
}
#[test]
fn reference_validation_and_resampling() -> Result<()> {
    assert!(ReferenceVoice::new(vec![0.0; 16000], 16000, None).is_err());
    assert!(ReferenceVoice::new(vec![0.1; 16000 * 16], 16000, None).is_err());
    assert!(ReferenceVoice::new(vec![f32::INFINITY; 16000], 16000, None).is_err());
    let audio = (0..16000).map(|n| (n as f32 * 0.04).sin() * 0.2).collect();
    let voice = ReferenceVoice::new(audio, 16000, Some("参考文本".into()))?;
    assert_eq!(voice.samples().len(), 24000);
    assert!(voice.samples().iter().all(|x| x.is_finite()));
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("立体声.wav");
    let mut wav = hound::WavWriter::create(
        &path,
        hound::WavSpec {
            channels: 2,
            sample_rate: 16000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )?;
    for _ in 0..16000 {
        wav.write_sample(1000i16)?;
        wav.write_sample(3000i16)?;
    }
    wav.finalize()?;
    let reference = ReferenceVoice::from_wav(path, None)?;
    assert!((reference.samples()[12000] - 2000.0 / 32768.0).abs() < 1e-4);
    Ok(())
}
