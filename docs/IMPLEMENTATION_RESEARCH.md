# Qwen3-TTS implementation research

Checked 2026-10-09. Target: Qwen3-TTS 12Hz 0.6B Base, Chinese and English,
reference-audio cloning, Windows/Linux/macOS, a Rust application API and no GPL
source. Repository claims are not equivalent to successful platform tests.

| Project | Declared license | Findings / decision |
|---|---|---|
| [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) | Apache-2.0 code and official weights | Authoritative model behavior. Python/PyTorch is valuable for comparison, but adds a large application runtime. Base is required for cloning; CustomVoice contains fixed speakers. |
| [ServeurpersoCom/qwentts.cpp](https://github.com/ServeurpersoCom/qwentts.cpp) | Full MIT LICENSE | Selected CPU core. Explicit C ABI, streaming callback, reference-latent extraction/reuse, UTF-8 Windows file handling, generation cancellation and matching preconverted Q8/Q4 models. Source and its ggml fork are pinned and included. Three-platform claims will be checked in our own CI. |
| [offgridai/qwen3-tts-cpp-streaming](https://github.com/offgridai/qwen3-tts-cpp-streaming) | Apache-2.0 | Alternative native streaming implementation. Public documentation reviewed; no source copied or linked. |
| [predict-woo/qwen3-tts.cpp](https://github.com/predict-woo/qwen3-tts.cpp) and [Danmoreng fork](https://github.com/Danmoreng/qwen3-tts.cpp) | MIT | Alternative GGML implementations, including Windows/Linux work. Public documentation reviewed; no source copied or linked. |
| [yet-another-ai/qts](https://github.com/yet-another-ai/qts) | Apache-2.0 | Rust frontend with ggml speech model and ONNX vocoder. A viable alternative but introduces two native inference runtimes. No code copied. |
| [second-state/qwen3_tts_rs](https://github.com/second-state/qwen3_tts_rs) | Apache-2.0 declared in Cargo.toml/README | Rust with libtorch or Apple-only MLX backend. Current audio-format dependencies and native distribution need additional review. Not used. |
| [TrevorS/qwen3-tts-rs](https://github.com/TrevorS/qwen3-tts-rs), published derivative `speakers-qwen3-tts` 0.3.0 | MIT declared in Cargo metadata | Pure Rust/Candle alternative. Published code currently constrains Candle to 0.9 and tokenizers to 0.22, and inspected packages lacked a full standalone LICENSE notice. Not adopted; prefer clear preserved notices and one validated runtime for the first version. |
| [0xShug0/audio.cpp](https://github.com/0xShug0/audio.cpp) | GitHub metadata returned NOASSERTION | Not adopted; not assumed permissive merely from a README or similar project. No implementation source copied. |

The selected path delivers a Rust library without embedding Python or making
runtime subprocess calls. It deliberately reuses a native model implementation;
a future pure Rust backend can implement the same TtsModel contract.

Q8 is the initial regression baseline; Q4 is cataloged as a smaller optional
variant. Full ICL cloning uses reference WAV plus matching transcript; x-vector
mode uses only the speaker identity. Long text is chunked in Rust, while native
inference streams short PCM blocks. Each CI platform must load real weights and
generate actual audio; compile-only checks are insufficient.

Read [THIRD_PARTY.md](../THIRD_PARTY.md) for exact revision evidence and licensing
scope. Python documentation/repositories listed above are research candidates,
not bundled runtime dependencies.
