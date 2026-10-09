#![cfg(all(feature = "qwen3", feature = "download"))]
use anyhow::{Result, ensure};
use serde_json::json;
use std::{
    path::{Path, PathBuf},
    time::Instant,
};
use valle_tts::{
    Language, ReferenceVoice, SynthesisOptions, TtsModel, WavOutput,
    cache::{ModelCache, builtin_model},
    models::qwen3::Qwen3,
};

#[test]
#[ignore = "downloads 1.28 GB of pinned Qwen3-TTS Q8 weights; CI runs explicitly"]
fn real_bilingual_voice_cloning_streaming_and_cancellation() -> Result<()> {
    let root = std::env::var_os("VALLE_TTS_TEST_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models"));
    let spec = builtin_model("qwen3-tts-0.6b-base-q8")?;
    let dir = ModelCache::new(root).ensure(&spec, true)?;
    // Exercise native Windows UTF-8 model paths without duplicating weights.
    let unicode_dir = dir.join("模型 路径");
    std::fs::create_dir_all(&unicode_dir)?;
    let mut paths = Vec::new();
    for file in &spec.files {
        let destination = unicode_dir.join(&file.path);
        if !destination.exists() {
            std::fs::hard_link(dir.join(&file.path), &destination)?;
        }
        paths.push(destination);
    }
    eprintln!("[test] loading {} on {}", spec.id, std::env::consts::OS);
    let mut model = Qwen3::load(&spec.id, &paths[0], &paths[1])?;
    eprintln!("[test] model loaded: {}", Qwen3::backend_version());
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let english = ReferenceVoice::from_wav(
        fixtures.join("sample1.wav"),
        Some(
            std::fs::read_to_string(fixtures.join("sample1.txt"))?
                .trim()
                .into(),
        ),
    )?;
    let chinese = ReferenceVoice::from_wav(
        fixtures.join("zh-reference.wav"),
        Some(
            std::fs::read_to_string(fixtures.join("zh-reference.txt"))?
                .trim()
                .into(),
        ),
    )?;
    let out = Path::new("artifacts");
    std::fs::create_dir_all(out)?;
    let mut receipts = Vec::new();
    let cases = [
        (
            "en-from-en",
            "The weather is lovely today.",
            Language::English,
            &english,
            160,
        ),
        (
            "zh-from-en",
            "你好，欢迎使用本地语音合成。",
            Language::Chinese,
            &english,
            160,
        ),
        (
            "zh-from-zh",
            "今天天气很好，我们一起出去散步。",
            Language::Chinese,
            &chinese,
            160,
        ),
        (
            "en-from-zh",
            "Welcome to our local voice assistant.",
            Language::English,
            &chinese,
            160,
        ),
        (
            "mixed",
            "你好，welcome to Valle，祝你今天愉快。",
            Language::Auto,
            &chinese,
            160,
        ),
        (
            "chunked",
            "你好，欢迎使用。这是第二句话。最后祝你愉快。",
            Language::Chinese,
            &chinese,
            12,
        ),
    ];
    for (name, text, language, voice, limit) in cases {
        eprintln!("[test] starting {name}");
        let start = Instant::now();
        let mut first = None;
        let mut callbacks = 0;
        let mut largest_chunk = 0;
        let mut sum_squares = 0f64;
        let mut peak = 0f32;
        let mut writer = WavOutput::new(out.join(format!("{name}.wav")), 24000)?;
        let options = SynthesisOptions {
            language,
            max_tokens: 160,
            max_chunk_chars: limit,
            ..Default::default()
        };
        let summary = model.synthesize(text, voice, &options, &mut |chunk| {
            first.get_or_insert(start.elapsed().as_millis());
            callbacks += 1;
            largest_chunk = largest_chunk.max(chunk.samples.len());
            for &s in &chunk.samples {
                sum_squares += f64::from(s).powi(2);
                peak = peak.max(s.abs());
            }
            writer.write(chunk)
        })?;
        writer.finish()?;
        let duration = summary.samples as f64 / 24000.0;
        let rms = (sum_squares / summary.samples as f64).sqrt();
        ensure!(
            duration > 0.3 && duration < 40.0,
            "unexpected audio duration for {name}: {duration}"
        );
        ensure!(
            rms > 0.001 && rms < 1.0 && peak > 0.01,
            "invalid/silent audio for {name}: rms={rms} peak={peak}"
        );
        ensure!(
            callbacks > 1 && largest_chunk <= 24000 * 3,
            "streaming did not emit bounded chunks"
        );
        if name == "chunked" {
            ensure!(summary.text_chunks > 1, "long text was not chunked");
        }
        let receipt = json!({"case":name,"text":text,"language":language,"samples":summary.samples,"duration_seconds":duration,
            "text_chunks":summary.text_chunks,"audio_callbacks":callbacks,"largest_audio_chunk":largest_chunk,
            "first_audio_ms":first,"wall_ms":start.elapsed().as_millis(),"rms":rms,"peak":peak});
        eprintln!("{receipt}");
        receipts.push(receipt);
    }
    // A larger generation budget must resize KV between requests, invalidate
    // old graph views, and keep both seeded audio and the handle reusable.
    // The short budget uses 256 positions; the larger one needs 768 or more.
    eprintln!("[test] checking KV growth and deterministic context reuse");
    let mut reference = Vec::new();
    for budget in [160, 700, 160] {
        let mut audio = Vec::new();
        model.synthesize(
            "今天天气很好，我们一起出去散步。",
            &chinese,
            &SynthesisOptions {
                language: Language::Chinese,
                max_tokens: budget,
                ..Default::default()
            },
            &mut |chunk| {
                audio.extend(chunk.samples);
                Ok(())
            },
        )?;
        if reference.is_empty() {
            reference = audio;
        } else {
            ensure!(
                audio == reference,
                "KV growth changed seeded audio or left stale graph views"
            );
        }
    }
    drop(reference);

    // Stop a live native worker after the first callback, then prove the context
    // remains reusable and that user sink failures return to Rust unchanged.
    eprintln!("[test] checking live cancellation");
    let token = valle_tts::CancellationToken::default();
    let callback_token = token.clone();
    let cancelled = model.synthesize(
        "这是一段需要取消的语音。",
        &chinese,
        &SynthesisOptions {
            language: Language::Chinese,
            cancellation: token,
            ..Default::default()
        },
        &mut |_| {
            callback_token.cancel();
            Ok(())
        },
    );
    ensure!(cancelled.is_err(), "cooperative cancellation was ignored");
    eprintln!("[test] checking sink error propagation and context reuse");
    let error = model
        .synthesize(
            "你好。",
            &chinese,
            &SynthesisOptions {
                language: Language::Chinese,
                ..Default::default()
            },
            &mut |_| anyhow::bail!("intentional sink failure"),
        )
        .unwrap_err();
    ensure!(
        error.to_string().contains("intentional sink failure"),
        "sink error was lost: {error}"
    );
    eprintln!("[test] checking generation budget failure");
    let budget = model.synthesize(
        "这句话不能在一帧之内读完。",
        &chinese,
        &SynthesisOptions {
            language: Language::Chinese,
            max_tokens: 1,
            ..Default::default()
        },
        &mut |_| Ok(()),
    );
    ensure!(
        budget.is_err(),
        "truncated generation was reported as success"
    );
    std::fs::write(
        out.join("validation.json"),
        serde_json::to_vec_pretty(&json!({
            "platform":std::env::consts::OS,"architecture":std::env::consts::ARCH,"model":spec.id,"revision":spec.revision,
            "native_backend":Qwen3::backend_version(),"cases":receipts,"kv_budget_growth_preserves_audio":true,"cancellation":true,"sink_error_propagation":true,"generation_budget_failure":true,
            "scope":"Real inference and audio integrity only; intelligibility and speaker similarity require separate evaluation."
        }))?,
    )?;
    Ok(())
}
