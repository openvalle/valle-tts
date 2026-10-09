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

```toml
[dependencies]
valle-tts = { version = "0.1.0", default-features = false, features = ["qwen3", "download"] }
```

Model weights are downloaded separately and are not embedded in the crate.

```rust,no_run
# #[cfg(feature = "qwen3")]
# fn main() -> anyhow::Result<()> {
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
# Ok(())
# }
# #[cfg(not(feature = "qwen3"))]
# fn main() {}
```

Output is streamed mono PCM at 24 kHz. A fallible `Send` sink receives owned
chunks, and may run on the native compute worker. Sink errors and panics return
to Rust without unwinding across FFI. Cooperative cancellation covers reference
preparation, prompt prefill, generation and codec compute.

Long input is split at punctuation/word boundaries with a Unicode character
limit (`--max-chunk-chars`, default 160). Inference retains one text chunk and
its bounded KV/codec state; WAV writing does not collect a whole output in RAM.
Single-request Talker KV is allocated when the prompt is known, in 256-position
classes covering prompt plus generation budget, up to 4096 positions. A larger
request releases old decode graphs and KV before allocating the replacement;
capacity is then reused. For this 0.6B model, 256/768/4096 positions use
56/168/896 MiB of Talker KV, excluding other weights, graphs and codec state.
CPU inference reads unconverted weights directly from the immutable GGUF
mapping. Only weights needing type conversion/layout changes use owned copies.
The input `&str` is owned by the caller; `--text-file` currently reads text into
memory. Independent chunk synthesis can change prosody at boundaries.
`--max-tokens` is a per-chunk hard frame budget (512 = 40.96 seconds). Exhausting
it before EOS fails instead of silently committing truncated audio. WAV output
uses a same-directory temporary file and replaces the destination only after
successful generation. Standard RIFF's 4 GiB limit still applies.

### Cancellation

Keep a clone of the request's `CancellationToken` in the UI or controlling
thread, and call `cancel()` to stop work:

```rust
use valle_tts::{CancellationToken, SynthesisOptions};

let stop = CancellationToken::default();
let options = SynthesisOptions {
    cancellation: stop.clone(),
    ..Default::default()
};
// Pass &options to the inference worker; keep stop in the controlling thread.
stop.cancel();
assert!(options.cancellation.is_cancelled());
```

`CancellationToken::from_shared_flag(Arc<AtomicBool>)` also accepts a host's
existing stop flag without a monitoring thread. The flag must only transition
from `false` to `true` during a request. Cancellation is permanent for all token
clones; use a fresh token for the next request.

Qwen checks cancellation between text chunks and generation frames and uses
GGML's CPU abort checkpoints during reference speaker/codec encoding, prompt
prefill and audio decoding. Native exceptions are contained, incomplete jobs
are detached, and borrowed stop callbacks are cleared before returning to Rust.

A cancelled request returns an error identifiable with
`error.is::<valle_tts::Cancelled>()`, including when error context is attached;
it never reports successful partial output. Already delivered sink chunks
cannot be retracted. Call a transactional output writer's `finish()` only after
successful inference, and reuse the backend with a fresh token after cancellation.
Custom backends must check the token during their own compute; the engine also
checks before dispatch, around sink callbacks and before returning success.

Cancellation is cooperative: a running tensor/CPU operator or blocking file
operation finishes before the next checkpoint. Model constructors and downloads
have no cancellation parameter. No fixed wall-clock cancellation latency is
promised, and no extra polling thread is created by the library.

### Features

- `qwen3`: the built-in Qwen3 CPU backend.
- `download`: revision-pinned model downloads with size and SHA-256 checks.
- `cli`: the command-line application, including `qwen3` and `download`.

All three are enabled by default. The model-independent core can be used with
`default-features = false` and no additional features.

## Validation

Three independent workflows run in parallel: [Linux](https://github.com/openvalle/valle-tts/actions/workflows/ci-linux.yml),
[Windows](https://github.com/openvalle/valle-tts/actions/workflows/ci-windows.yml), [macOS](https://github.com/openvalle/valle-tts/actions/workflows/ci-macos.yml).
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
are Apache-2.0. See [THIRD_PARTY.md](https://github.com/openvalle/valle-tts/blob/main/THIRD_PARTY.md) for exact source revisions,
copied/modified files, fixture origins and preserved notices. No GPL-family
code is used in the enabled implementation. The initial implementation research
is in [docs/IMPLEMENTATION_RESEARCH.md](https://github.com/openvalle/valle-tts/blob/main/docs/IMPLEMENTATION_RESEARCH.md).

## Crate development and release checks

The crate targets Rust 1.99+ and edition 2024. Run formatting, Clippy, unit/API
tests, documentation and packaging checks before a release. Public APIs must
document units, callback execution, resource ownership and failure conditions;
missing public documentation and broken Rustdoc links fail validation. The README
provides the crate-level documentation, and its Rust example is compiled as a
documentation test.
Changes to native sources must retain upstream notices, source hashes and
reproducible patches. The permissive-license policy in THIRD_PARTY.md applies.

The default features are `qwen3`, `download` and `cli`. Library consumers can use
`default-features = false` with `features = ["qwen3"]` for local model directories,
or add `download` for the verified cache. No-default-features builds expose the
model-independent core. The `cli` feature enables the backend and downloader;
Clap is excluded from builds that omit `cli`. TTS's `qwen3` feature additionally needs CMake
and a C++17 compiler. Disabling it builds the TTS core without native compilation.

Each independent Windows/Linux/macOS workflow checks supported feature
combinations, warning-free API documentation and `cargo publish --dry-run`,
which extracts and builds the exact crate archive without uploading it.
Packaging explicitly includes source, model catalogs, test fixtures and third
party notices; it also includes its complete pinned native build sources.
Model weights, generated audio, local artifacts and build caches are excluded.

```sh
cargo fmt --all -- --check
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo test --locked
cargo test --locked --no-default-features
cargo doc --locked --no-default-features --features qwen3,download --no-deps --lib
cargo package --locked --list
cargo publish --locked --dry-run
```

The initial API is version 0.1.0. Breaking public API changes require a minor
version increment before 1.0; compatible fixes use a patch increment. Document
changes in the release notes and tag each published version. The minimum Rust
version is declared in `Cargo.toml` and matches the toolchain used in CI.

Before publishing, require all three platform workflows (including the explicit
real-model tests) to pass, review the archive's contents and licenses, and confirm
the version is new on crates.io. A dry run does not reserve the crate name or
verify the publisher account's ownership. Publishing a GitHub Release tagged `v<version>` triggers
`.github/workflows/release.yml`. The tag must match `Cargo.toml` and belong to
main history; all three independent platform workflows must have passed at
that exact commit. The workflow uses the repository secret
`CARGO_REGISTRY_TOKEN`, validates the archive before uploading, and checks its
SHA-256 against crates.io. Re-running an identical published archive is safe;
an existing version with different contents is rejected. Manual dispatch can
validate an existing tag with `publish: false` before uploading it.

## 0.1.0 release notes

- Initial Rust library and CLI with pluggable TTS backends.
- Qwen3-TTS 0.6B Base Q8/Q4 voice cloning in Chinese and English.
- Streamed PCM/WAV output, cancellation, and bounded per-chunk generation.
- Revision-pinned, verified model downloads and permissively licensed sources.
- Independent Linux, Windows and macOS CI with real-model and crate checks.
- Shared host cancellation flags, typed cancellation errors and reusable backends.
