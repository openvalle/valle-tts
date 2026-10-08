use anyhow::{Context, Result, ensure};
use std::path::Path;

/// A short mono voice reference, resampled to the model-independent 24 kHz contract.
#[derive(Debug, Clone, PartialEq)]
pub struct ReferenceVoice {
    pub(crate) samples: Vec<f32>,
    pub(crate) transcript: Option<String>,
}
impl ReferenceVoice {
    pub fn new(samples: Vec<f32>, sample_rate: u32, transcript: Option<String>) -> Result<Self> {
        ensure!(
            (8000..=192000).contains(&sample_rate),
            "reference sample rate must be 8..=192 kHz"
        );
        ensure!(
            samples.len() >= sample_rate as usize / 2 && samples.len() <= sample_rate as usize * 15,
            "reference must be 0.5..=15 seconds; use a clean 3..=10 second clip"
        );
        ensure!(
            samples.iter().all(|x| x.is_finite()),
            "reference contains non-finite audio"
        );
        ensure!(
            samples.iter().any(|x| x.abs() > 1e-5),
            "reference is silent"
        );
        if let Some(text) = &transcript {
            ensure!(
                !text.trim().is_empty() && text.chars().count() <= 1024 && !text.contains('\0'),
                "reference transcript is empty, too long or contains NUL"
            );
        }
        let samples = resample(&samples, sample_rate, 24000);
        Ok(Self {
            samples,
            transcript,
        })
    }
    pub fn from_wav(path: impl AsRef<Path>, transcript: Option<String>) -> Result<Self> {
        let mut reader = hound::WavReader::open(path.as_ref())
            .with_context(|| format!("read reference {}", path.as_ref().display()))?;
        let spec = reader.spec();
        ensure!(
            (1..=8).contains(&spec.channels),
            "reference WAV must have 1..=8 channels"
        );
        ensure!(
            (8000..=192000).contains(&spec.sample_rate),
            "reference WAV has unsupported sample rate"
        );
        ensure!(
            reader.duration() <= spec.sample_rate * 15,
            "reference exceeds 15 seconds"
        );
        let values: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
            hound::SampleFormat::Int => {
                let scale = 2f32.powi(i32::from(spec.bits_per_sample) - 1);
                reader
                    .samples::<i32>()
                    .map(|s| s.map(|v| v as f32 / scale))
                    .collect::<Result<_, _>>()?
            }
        };
        let channels = usize::from(spec.channels);
        ensure!(
            values.len().is_multiple_of(channels),
            "incomplete reference channel frame"
        );
        let mono = values
            .chunks_exact(channels)
            .map(|v| v.iter().sum::<f32>() / channels as f32)
            .collect();
        Self::new(mono, spec.sample_rate, transcript)
    }
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }
    pub fn transcript(&self) -> Option<&str> {
        self.transcript.as_deref()
    }
}

// Windowed sinc with a low-pass cutoff for downsampling. Reference audio is bounded.
fn resample(samples: &[f32], source: u32, target: u32) -> Vec<f32> {
    if source == target {
        return samples.to_vec();
    }
    let len = (samples.len() as u64 * u64::from(target) / u64::from(source)) as usize;
    let ratio = f64::from(source) / f64::from(target);
    let cutoff = (f64::from(target) / f64::from(source)).min(1.0) * 0.94;
    let radius = (24.0 / cutoff).ceil() as i64;
    (0..len)
        .map(|i| {
            let position = i as f64 * ratio;
            let center = position.floor() as i64;
            let mut sum = 0.0;
            let mut weight = 0.0;
            for offset in -radius..=radius {
                let index = center + offset;
                if index < 0 || index >= samples.len() as i64 {
                    continue;
                }
                let distance = position - index as f64;
                let x = std::f64::consts::PI * distance * cutoff;
                let sinc = if x.abs() < 1e-10 { 1.0 } else { x.sin() / x };
                let window = 0.5 + 0.5 * (std::f64::consts::PI * distance / radius as f64).cos();
                let w = cutoff * sinc * window;
                sum += f64::from(samples[index as usize]) * w;
                weight += w;
            }
            (sum / weight) as f32
        })
        .collect()
}
