use crate::AudioChunk;
use anyhow::{Result, ensure};
use std::{fs::File, io::BufWriter, path::Path};

/// Streams PCM16 to a temporary WAV. Only finish() replaces the destination.
pub struct WavOutput {
    writer: Option<hound::WavWriter<BufWriter<File>>>,
    temporary: tempfile::NamedTempFile,
    destination: std::path::PathBuf,
    sample_rate: u32,
    samples: u64,
}
impl WavOutput {
    pub fn new(path: impl AsRef<Path>, sample_rate: u32) -> Result<Self> {
        ensure!(sample_rate > 0, "sample rate is zero");
        let path = path.as_ref();
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let temporary = tempfile::NamedTempFile::new_in(parent)?;
        let file = temporary.reopen()?;
        let writer = hound::WavWriter::new(
            BufWriter::new(file),
            hound::WavSpec {
                channels: 1,
                sample_rate,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )?;
        Ok(Self {
            writer: Some(writer),
            temporary,
            destination: path.to_owned(),
            sample_rate,
            samples: 0,
        })
    }
    pub fn write(&mut self, chunk: AudioChunk) -> Result<()> {
        ensure!(
            chunk.sample_rate == self.sample_rate,
            "backend changed sample rate"
        );
        ensure!(
            chunk.samples.iter().all(|x| x.is_finite()),
            "backend produced non-finite audio"
        );
        let next = self
            .samples
            .checked_add(chunk.samples.len() as u64)
            .ok_or_else(|| anyhow::anyhow!("WAV sample count overflow"))?;
        ensure!(
            next <= (u32::MAX as u64 - 64) / 2,
            "output exceeds RIFF WAV limit; use multiple output files"
        );
        let writer = self.writer.as_mut().expect("writer available until finish");
        for s in chunk.samples {
            writer.write_sample((s.clamp(-1.0, 1.0) * 32767.0).round() as i16)?;
        }
        self.samples = next;
        Ok(())
    }
    pub fn finish(mut self) -> Result<()> {
        ensure!(self.samples > 0, "model produced no audio");
        self.writer.take().expect("writer available").finalize()?;
        self.temporary.as_file().sync_all()?;
        self.temporary.persist(&self.destination)?;
        Ok(())
    }
}
