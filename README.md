# valle-tts

Local Chinese/English voice cloning with a Rust library and CLI. The first model
is **Qwen3-TTS 12Hz 0.6B Base**, with revision-pinned Q8 and Q4 GGUF downloads.
The API accepts additional model families through `TtsModel`.

The Qwen backend statically links a pinned MIT-licensed GGML C++ core. This is
**not a pure Rust inference implementation**. No Python, libtorch, external TTS
process, network service or system-installed model runtime is needed at runtime.
CPU is the initial common backend for Windows x64, Linux x64 and macOS ARM64.
The x64 builds require AVX2, FMA and F16C. GPU acceleration and other CPU
architectures are not yet validated.

## Build

Install the Rust toolchain from `rust-toolchain.toml`, CMake 3.20+ and a C/C++17
compiler. On Windows install Visual Studio Build Tools with Desktop development
with C++; on macOS install Xcode command line tools; on Linux use GCC or Clang.

```sh
cargo build --locked --release
cargo run --locked --release -- models
cargo run --locked --release -- download --cache-dir models
```

Direct crates were checked against the latest stable crates.io releases on
2026-10-09. Rust is pinned to 1.99.0; Cargo.lock also pins transitive versions.
Native sources are committed, so building never fetches C++ dependencies.
Native inference uses Release optimization even in `cargo test`; MSVC's
Release flags explicitly retain optimization and C++ exception handling.

## Voice cloning

Use a clean 3–10 second WAV reference and its matching transcript. Input WAV
may be mono or stereo and is mixed down and resampled to 24 kHz. Accepted
reference duration is 0.5–15 seconds, sample rate 8–192 kHz, up to 8 channels.

```sh
cargo run --locked --release -- synthesize \
  --cache-dir models --offline \
  --reference tests/fixtures/zh-reference.wav \
  --ref-text-file tests/fixtures/zh-reference.txt \
  --text "你好，欢迎使用本地语音合成。" --language zh --output chinese.wav

cargo run --locked --release -- synthesize \
  --cache-dir models --offline \
  --reference tests/fixtures/zh-reference.wav \
  --ref-text-file tests/fixtures/zh-reference.txt \
  --text "Welcome to our local voice assistant." --language en --output english.wav
```

`--language auto` allows mixed Chinese/English text. `--x-vector-only` selects
speaker-embedding conditioning without a transcript; full in-context cloning
uses the transcript and reference codec codes. Only Base checkpoints are
accepted for voice cloning. The API caches the most recently prepared reference
and reuses its latents across text chunks and synthesis calls.

The default model is `qwen3-tts-0.6b-base-q8` (about 1.28 GB including its Q8
codec). Use `--model qwen3-tts-0.6b-base-q4` for the 0.92 GB alternative; Q4
quality is not assumed equivalent to Q8. These are download sizes, not peak RAM.

## Library and streaming

```rust,ignore
use valle_tts::{ReferenceVoice, SynthesisOptions, TtsEngine, WavOutput};
use valle_tts::models::qwen3::Qwen3;

let model = Qwen3::load("qwen3-tts-0.6b-base-q8", "talker.gguf", "codec.gguf")?;
let voice = ReferenceVoice::from_wav("reference.wav", Some("Reference transcript".into()))?;
let mut engine = TtsEngine::new();
engine.register(model)?;
let mut wav = WavOutput::new("output.wav", 24000)?;
let summary = engine.synthesize("qwen3-tts-0.6b-base-q8", "Hello, 你好！", &voice,
    &SynthesisOptions::default(), &mut |chunk| wav.write(chunk))?;
wav.finish()?;
```

Output is streamed mono PCM at 24 kHz. A fallible `Send` sink receives owned
chunks, and may run on the native compute worker. Sink errors and panics return
to Rust without unwinding across FFI. `CancellationToken` cooperatively stops
active generation at frame boundaries. Reference preprocessing itself is not
interruptible.

Long input is split at punctuation/word boundaries with a Unicode character
limit (`--max-chunk-chars`, default 160). Inference retains one text chunk and
its bounded KV/codec state; WAV writing does not collect a whole output in RAM.
The input `&str` is owned by the caller; `--text-file` currently reads text into
memory. Independent chunk synthesis can change prosody at boundaries.
`--max-tokens` is a per-chunk hard frame budget (512 = 40.96 seconds). Exhausting
it before EOS fails instead of silently committing truncated audio. WAV output
uses a same-directory temporary file and replaces the destination only after
successful generation. Standard RIFF's 4 GiB limit still applies.

## Validation

Three independent workflows run in parallel: [Linux](.github/workflows/ci-linux.yml),
[Windows](.github/workflows/ci-windows.yml), [macOS](.github/workflows/ci-macos.yml).
Each runs formatting, Clippy, license/provenance checks, optional-backend checks,
API tests and **explicit real-model inference**; a skipped model test is not a pass.
Rust/native build caches are platform-specific. Verified model weights use a
shared cross-OS cache keyed by the full pinned catalog.

```sh
cargo test --locked
python3 scripts/check_vendor.py
python3 scripts/check_licenses.py
cargo run --locked -- download --cache-dir models
cargo test --locked --test real_models -- --ignored --nocapture
```

Real tests cover English, Chinese, both directions of cross-language cloning,
mixed text, multiple text chunks, UTF-8 model paths, cancellation, generation
budget failures and sink errors. Generated WAV files and machine-readable
`artifacts/validation.json` are uploaded by CI. Audio integrity and inference
success do not establish pronunciation accuracy or speaker similarity; those
need separate ASR and listening evaluations. Physical Windows hardware testing
remains useful after CI passes.

## Licenses

New Valle code is Apache-2.0. Copied native sources remain MIT, and model weights
are Apache-2.0. See [THIRD_PARTY.md](THIRD_PARTY.md) for exact source revisions,
copied/modified files, fixture origins and preserved notices. No GPL-family
code is used in the enabled implementation. The initial implementation research
is in [docs/IMPLEMENTATION_RESEARCH.md](docs/IMPLEMENTATION_RESEARCH.md).
