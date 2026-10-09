//! Qwen3-TTS Base on a pinned, statically linked MIT GGML implementation.
use crate::{
    AudioChunk, CancellationToken, ModelInfo, ReferenceVoice, SynthesisOptions, SynthesisSummary,
    TtsModel,
};
use anyhow::{Context, Result, ensure};
use std::{
    ffi::{CStr, CString, c_char, c_void},
    path::Path,
    ptr::NonNull,
};

#[repr(C)]
struct Request {
    text: *const c_char,
    language: *const c_char,
    transcript: *const c_char,
    voice: *const c_void,
    seed: i64,
    max_tokens: i32,
    temperature: f32,
    emit: unsafe extern "C" fn(*const f32, i32, *mut c_void) -> bool,
    cancel: unsafe extern "C" fn(*mut c_void) -> bool,
    cancel_user_data: *mut c_void,
    user_data: *mut c_void,
}
unsafe extern "C" {
    fn vt_load(talker: *const c_char, codec: *const c_char) -> *mut c_void;
    fn vt_free(ctx: *mut c_void);
    fn vt_error() -> *const c_char;
    fn vt_model_type(ctx: *mut c_void) -> *const c_char;
    fn vt_version() -> *const c_char;
    fn vt_voice(
        ctx: *mut c_void,
        samples: *const f32,
        count: i32,
        cancel: unsafe extern "C" fn(*mut c_void) -> bool,
        cancel_data: *mut c_void,
    ) -> *mut c_void;
    fn vt_voice_free(voice: *mut c_void);
    fn vt_synthesize(ctx: *mut c_void, request: *const Request) -> i32;
}

struct NativeContext(NonNull<c_void>);
impl Drop for NativeContext {
    fn drop(&mut self) {
        unsafe {
            vt_free(self.0.as_ptr());
        }
    }
}
// qt_synthesize blocks until its worker finishes all callbacks. An owned handle
// can move between threads; &mut Qwen3 serializes all uses. No Sync is exposed.
unsafe impl Send for NativeContext {}
struct NativeVoice(NonNull<c_void>);
impl Drop for NativeVoice {
    fn drop(&mut self) {
        unsafe {
            vt_voice_free(self.0.as_ptr());
        }
    }
}
// This owns immutable malloc buffers; only &mut Qwen3 accesses or releases them.
unsafe impl Send for NativeVoice {}

/// Qwen3-TTS Base using the statically linked GGML CPU backend.
/// The handle is `Send`, and `&mut self` serializes synthesis and reference caching.
pub struct Qwen3 {
    context: NativeContext,
    prepared: Option<(ReferenceVoice, NativeVoice)>,
    id: String,
}
impl Qwen3 {
    /// Load local Talker and codec GGUF files for a supported 0.6B Base model ID.
    /// No files are downloaded. Build-time CMake/C++ requirements are documented
    /// in the crate README; the resulting native runtime is linked statically.
    ///
    /// # Errors
    /// Rejects unsupported IDs, invalid paths, failed model loading and non-Base checkpoints.
    pub fn load(id: &str, talker: impl AsRef<Path>, codec: impl AsRef<Path>) -> Result<Self> {
        ensure!(
            matches!(id, "qwen3-tts-0.6b-base-q8" | "qwen3-tts-0.6b-base-q4"),
            "unsupported Qwen model ID: {id}"
        );
        let talker = path_string(talker.as_ref())?;
        let codec = path_string(codec.as_ref())?;
        // Both strings remain alive throughout the synchronous native load.
        let context = NativeContext(
            NonNull::new(unsafe { vt_load(talker.as_ptr(), codec.as_ptr()) })
                .context(native_error())?,
        );
        let kind = unsafe { CStr::from_ptr(vt_model_type(context.0.as_ptr())) }.to_str()?;
        ensure!(
            kind == "base",
            "voice cloning requires a Base checkpoint, found {kind}"
        );
        Ok(Self {
            context,
            prepared: None,
            id: id.to_owned(),
        })
    }
    /// Return the pinned native backend version used by this build.
    pub fn backend_version() -> String {
        unsafe { CStr::from_ptr(vt_version()) }
            .to_string_lossy()
            .into_owned()
    }
    fn prepare(&mut self, voice: &ReferenceVoice, cancellation: &CancellationToken) -> Result<()> {
        cancellation.check()?;
        if self.prepared.as_ref().is_some_and(|(v, _)| v == voice) {
            return Ok(());
        }
        let count = i32::try_from(voice.samples.len())?;
        let native = unsafe {
            vt_voice(
                self.context.0.as_ptr(),
                voice.samples.as_ptr(),
                count,
                on_cancel,
                (cancellation as *const CancellationToken).cast_mut().cast(),
            )
        };
        // Own any completed reference before checking cancellation so a late
        // stop request cannot leak its native buffers or poison the old cache.
        let native = NonNull::new(native).map(NativeVoice);
        cancellation.check()?;
        let native = native.context(native_error())?;
        self.prepared = Some((voice.clone(), native));
        Ok(())
    }
}

impl TtsModel for Qwen3 {
    fn info(&self) -> ModelInfo {
        ModelInfo {
            id: self.id.clone(),
            family: "qwen3-tts".into(),
            languages: vec![crate::Language::Chinese, crate::Language::English],
            voice_cloning: true,
            sample_rate: 24000,
        }
    }
    fn synthesize(
        &mut self,
        text: &str,
        voice: &ReferenceVoice,
        options: &SynthesisOptions,
        emit: &mut (dyn FnMut(AudioChunk) -> Result<()> + Send),
    ) -> Result<SynthesisSummary> {
        options.validate()?;
        ensure!(!text.trim().is_empty(), "text is empty");
        ensure!(!text.contains('\0'), "text contains NUL");
        self.prepare(voice, &options.cancellation)?;
        let language = CString::new(options.language.name())?;
        let transcript = voice.transcript.as_deref().map(CString::new).transpose()?;
        let mut state = CallbackState {
            emit,
            cancellation: &options.cancellation,
            error: None,
            samples: 0,
        };
        let mut chunks = 0;
        for part in TextChunks::new(text, options.max_chunk_chars) {
            options.cancellation.check()?;
            let part = CString::new(part)?;
            let request = Request {
                text: part.as_ptr(),
                language: language.as_ptr(),
                transcript: transcript.as_ref().map_or(std::ptr::null(), |v| v.as_ptr()),
                voice: self.prepared.as_ref().expect("prepared voice").1.0.as_ptr(),
                seed: options.seed,
                max_tokens: options.max_tokens as i32,
                temperature: options.temperature,
                emit: on_audio,
                cancel: on_cancel,
                cancel_user_data: (&options.cancellation as *const CancellationToken)
                    .cast_mut()
                    .cast(),
                user_data: (&mut state as *mut CallbackState<'_>).cast(),
            };
            // Native waits for its compute worker before returning. All pointers
            // remain valid, and CallbackState has one exclusive worker at a time.
            let before = state.samples;
            let status = unsafe { vt_synthesize(self.context.0.as_ptr(), &request) };
            if let Some(error) = state.error.take() {
                return Err(error);
            }
            options.cancellation.check()?;
            ensure!(
                status == 0,
                "Qwen synthesis failed ({status}): {}",
                native_error()
            );
            ensure!(
                state.samples > before,
                "Qwen generated no audio for a text chunk"
            );
            chunks += 1;
        }
        options.cancellation.check()?;
        ensure!(state.samples > 0, "Qwen generated no audio");
        Ok(SynthesisSummary {
            model: self.id.clone(),
            sample_rate: 24000,
            samples: state.samples,
            text_chunks: chunks,
        })
    }
}
fn path_string(path: &Path) -> Result<CString> {
    ensure!(
        path.is_file(),
        "model file does not exist: {}",
        path.display()
    );
    CString::new(path.to_str().context("model path must be UTF-8")?).map_err(Into::into)
}
fn native_error() -> String {
    let ptr = unsafe { vt_error() };
    if ptr.is_null() {
        "native allocation/load failed".into()
    } else {
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    }
}
struct CallbackState<'a> {
    emit: &'a mut (dyn FnMut(AudioChunk) -> Result<()> + Send),
    cancellation: &'a crate::CancellationToken,
    error: Option<anyhow::Error>,
    samples: u64,
}
unsafe extern "C" fn on_cancel(user: *mut c_void) -> bool {
    // Native reads only a shared atomic token, never the mutable audio sink.
    // The synchronous call joins its worker and clears the callback before return.
    unsafe { &*(user.cast::<CancellationToken>()) }.is_cancelled()
}
unsafe extern "C" fn on_audio(samples: *const f32, count: i32, user: *mut c_void) -> bool {
    let state = unsafe { &mut *(user.cast::<CallbackState<'_>>()) };
    if state.cancellation.is_cancelled() || state.error.is_some() {
        return false;
    }
    if count <= 0 || samples.is_null() {
        state.error = Some(anyhow::anyhow!("native emitted an invalid audio buffer"));
        return false;
    }
    // The callback owns this borrow until it returns. Copy before invoking user
    // code; a sink may retain the chunk but must never retain a native pointer.
    let data = unsafe { std::slice::from_raw_parts(samples, count as usize) };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ensure!(
            data.iter().all(|x| x.is_finite()),
            "native produced non-finite audio"
        );
        (state.emit)(AudioChunk {
            samples: data.to_vec(),
            sample_rate: 24000,
        })
    }));
    match result {
        Ok(Ok(())) => {
            state.samples += count as u64;
            !state.cancellation.is_cancelled()
        }
        Ok(Err(error)) => {
            state.error = Some(error);
            false
        }
        Err(_) => {
            state.error = Some(anyhow::anyhow!("audio sink panicked"));
            false
        }
    }
}

/// Iterates borrowed text slices without collecting an entire long document.
struct TextChunks<'a> {
    remaining: &'a str,
    limit: usize,
}
impl<'a> TextChunks<'a> {
    fn new(text: &'a str, limit: usize) -> Self {
        Self {
            remaining: text,
            limit,
        }
    }
}
impl<'a> Iterator for TextChunks<'a> {
    type Item = &'a str;
    fn next(&mut self) -> Option<Self::Item> {
        self.remaining = self.remaining.trim_start();
        if self.remaining.is_empty() {
            return None;
        }
        let mut end = self.remaining.len();
        let mut boundary = None;
        let mut whitespace = None;
        for (n, (i, c)) in self.remaining.char_indices().enumerate() {
            if n == self.limit {
                end = i;
                break;
            }
            if matches!(c, '。' | '！' | '？' | '\n' | '.' | '!' | '?' | ';' | '；') {
                boundary = Some(i + c.len_utf8());
            }
            if c.is_whitespace() {
                whitespace = Some(i + c.len_utf8());
            }
        }
        if end < self.remaining.len() {
            end = boundary.or(whitespace).unwrap_or(end);
        }
        let (part, rest) = self.remaining.split_at(end);
        self.remaining = rest;
        Some(part)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "download")]
    #[test]
    #[ignore = "uses pinned Q8 weights; CI invokes this explicitly after warming its model cache"]
    fn native_preparation_and_prefill_cancel_and_context_reuse() -> Result<()> {
        use crate::cache::{ModelCache, builtin_model};
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct Stop {
            token: CancellationToken,
            polls: AtomicUsize,
            emitted: AtomicUsize,
            limit: AtomicUsize,
        }
        unsafe extern "C" fn cancel_after_compute_starts(user: *mut c_void) -> bool {
            let state = unsafe { &*user.cast::<Stop>() };
            if state.polls.fetch_add(1, Ordering::Relaxed) >= state.limit.load(Ordering::Relaxed) {
                state.token.cancel();
            }
            state.token.is_cancelled()
        }
        unsafe extern "C" fn count_audio(_: *const f32, _: i32, user: *mut c_void) -> bool {
            unsafe { &*user.cast::<Stop>() }
                .emitted
                .fetch_add(1, Ordering::Relaxed);
            true
        }
        unsafe extern "C" fn arm_stop_after_audio(
            _: *const f32,
            _: i32,
            user: *mut c_void,
        ) -> bool {
            let state = unsafe { &*user.cast::<Stop>() };
            if state.emitted.fetch_add(1, Ordering::Relaxed) == 0 {
                // Keep the token active until the next CPU graph has started,
                // instead of cancelling directly inside the sink callback.
                state
                    .limit
                    .store(state.polls.load(Ordering::Relaxed) + 1, Ordering::Relaxed);
            }
            true
        }
        let stop = || Stop {
            token: CancellationToken::default(),
            polls: AtomicUsize::new(0),
            emitted: AtomicUsize::new(0),
            limit: AtomicUsize::new(1),
        };
        let root = std::env::var_os("VALLE_TTS_TEST_CACHE")
            .context("VALLE_TTS_TEST_CACHE must point to verified models")?;
        let spec = builtin_model("qwen3-tts-0.6b-base-q8")?;
        let dir = ModelCache::new(root).ensure(&spec, true)?;
        let mut model = Qwen3::load(
            &spec.id,
            dir.join(&spec.files[0].path),
            dir.join(&spec.files[1].path),
        )?;
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let voice = ReferenceVoice::from_wav(
            fixtures.join("sample1.wav"),
            Some(
                std::fs::read_to_string(fixtures.join("sample1.txt"))?
                    .trim()
                    .to_owned(),
            ),
        )?;

        let preparation = stop();
        let extracted = unsafe {
            vt_voice(
                model.context.0.as_ptr(),
                voice.samples.as_ptr(),
                voice.samples.len() as i32,
                cancel_after_compute_starts,
                (&preparation as *const Stop).cast_mut().cast(),
            )
        };
        let extracted = NonNull::new(extracted).map(NativeVoice);
        ensure!(
            extracted.is_none() && preparation.token.is_cancelled(),
            "reference graph ignored cancellation"
        );
        ensure!(
            preparation.polls.load(Ordering::Relaxed) >= 2,
            "reference cancellation happened before CPU compute"
        );
        eprintln!("[test] reference CPU graph cancelled after compute started");

        let active = CancellationToken::default();
        let reference = unsafe {
            vt_voice(
                model.context.0.as_ptr(),
                voice.samples.as_ptr(),
                voice.samples.len() as i32,
                on_cancel,
                (&active as *const CancellationToken).cast_mut().cast(),
            )
        };
        let reference = NativeVoice(NonNull::new(reference).context(native_error())?);
        let prefill = stop();
        let text = CString::new("The quick brown fox jumps over the lazy dog.")?;
        let language = CString::new("english")?;
        let request = Request {
            text: text.as_ptr(),
            language: language.as_ptr(),
            transcript: std::ptr::null(),
            voice: reference.0.as_ptr(),
            seed: 42,
            max_tokens: 256,
            temperature: 0.9,
            emit: count_audio,
            cancel: cancel_after_compute_starts,
            cancel_user_data: (&prefill as *const Stop).cast_mut().cast(),
            user_data: (&prefill as *const Stop).cast_mut().cast(),
        };
        let status = unsafe { vt_synthesize(model.context.0.as_ptr(), &request) };
        ensure!(
            status != 0 && prefill.token.is_cancelled(),
            "prefill graph ignored cancellation"
        );
        ensure!(
            prefill.polls.load(Ordering::Relaxed) >= 2
                && prefill.emitted.load(Ordering::Relaxed) == 0,
            "cancellation must stop compute before first audio"
        );
        eprintln!("[test] prefill CPU graph cancelled before first audio");

        let generation = stop();
        generation.limit.store(usize::MAX, Ordering::Relaxed);
        let request = Request {
            emit: arm_stop_after_audio,
            cancel_user_data: (&generation as *const Stop).cast_mut().cast(),
            user_data: (&generation as *const Stop).cast_mut().cast(),
            ..request
        };
        let status = unsafe { vt_synthesize(model.context.0.as_ptr(), &request) };
        ensure!(
            status != 0 && generation.token.is_cancelled(),
            "generation CPU graph ignored cancellation"
        );
        ensure!(
            generation.emitted.load(Ordering::Relaxed) == 1,
            "audio continued after graph cancellation"
        );
        eprintln!("[test] active generation CPU graph cancelled after the first audio chunk");

        let result = model.synthesize(
            "Hello.",
            &voice,
            &SynthesisOptions::default(),
            &mut |chunk| {
                ensure!(
                    chunk.samples.iter().all(|sample| sample.is_finite()),
                    "invalid reused output"
                );
                Ok(())
            },
        )?;
        ensure!(result.samples > 0, "cancelled context was not reusable");
        eprintln!("[test] native context successfully reused with a fresh token");
        Ok(())
    }
    #[test]
    fn unicode_chunking_keeps_every_character_in_order() {
        let text = "你好，Valle！这是中英混读测试。Hello world! 再见。";
        let chunks: Vec<_> = TextChunks::new(text, 12).collect();
        assert!(chunks.iter().all(|c| c.chars().count() <= 12));
        assert_eq!(chunks.concat().replace(' ', ""), text.replace(' ', ""));
        assert!(chunks.len() > 1);
    }
    #[test]
    fn sink_panics_and_errors_do_not_cross_the_c_abi() {
        let token = crate::CancellationToken::default();
        let audio = [0.1f32; 4];
        for panic in [false, true] {
            let mut sink = |_: AudioChunk| -> Result<()> {
                assert!(!panic, "test sink panic");
                anyhow::bail!("disk full")
            };
            let mut state = CallbackState {
                emit: &mut sink,
                cancellation: &token,
                error: None,
                samples: 0,
            };
            assert!(!unsafe {
                on_audio(
                    audio.as_ptr(),
                    4,
                    (&mut state as *mut CallbackState<'_>).cast(),
                )
            });
            assert!(state.error.is_some());
            assert_eq!(state.samples, 0);
        }
    }
}
