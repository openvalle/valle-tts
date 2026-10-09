# Third-party provenance and licenses

This repository has a new Rust API, model dispatch, CLI, audio frontend, WAV
writer and safe native adapter. **The Qwen inference core is copied and linked
C++ source, not a new pure Rust neural network implementation.** Its existing
MIT license and notices are preserved. This distinction is intentional.

| Source | Pinned revision | License | Exact use |
|---|---|---|---|
| [ServeurpersoCom/qwentts.cpp](https://github.com/ServeurpersoCom/qwentts.cpp/tree/51512f129a7419567f4b8abfb06801451789b8f1) | `51512f129a7419567f4b8abfb06801451789b8f1` | MIT | `src/` copied to `vendor/qwentts/src/`; the three pipeline translation units are compiled and statically linked. No CLI, HTTP server, converter or third-party HTTP/JSON library is built. Upstream LICENSE retained. |
| [ServeurpersoCom/ggml](https://github.com/ServeurpersoCom/ggml/tree/2eddaf94a8cdbbf1c30017d28313cd661a9da543) | `2eddaf94a8cdbbf1c30017d28313cd661a9da543` | MIT | The upstream's exact ggml submodule, copied under `vendor/ggml/` without Git metadata, examples, tests, CI, docs and non-CPU backend source directories. Only CPU/core code is compiled; GPU/third-party acceleration paths are disabled. LICENSE retained. |
| [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) | Model revisions below | Apache-2.0 | Official architecture/model documentation reference; no Python source copied into our Rust code. Full license in `third_party/QwenLM__Qwen3-TTS-LICENSE`. |
| [openvalle/valle-asr](https://github.com/openvalle/valle-asr/tree/8d37bb1dffa84ee8610112c23f9a6f4912386f35) | `8d37bb1dffa84ee8610112c23f9a6f4912386f35` | Apache-2.0 | `src/cache.rs` and `scripts/check_licenses.py` adapted by replacing project/environment names; toolchain and independent CI organization reused. The common Apache-2.0 license is at repository root. |
| [alan890104/qwen3-asr-rs](https://github.com/alan890104/qwen3-asr-rs/tree/c5ef09646af6278d2ba8b8ceaf543ffb32d1a5dc) | `c5ef09646af6278d2ba8b8ceaf543ffb32d1a5dc` | MIT | English `tests/fixtures/sample1.wav` and transcript copied via valle-asr. Original source hashes are in `tests/fixtures/provenance.json`; only sample1 is included here. Full MIT notice retained under `third_party/`. |
| [AISHELL-1](https://www.openslr.org/33/) | Mirror `2724409d538167445e43ebf846990319f12a1cbf` | Apache-2.0 | One Chinese reference WAV and its original transcript. Source path, hash and pinned mirror are in `tests/fixtures/zh-provenance.json`; license in `third_party/AISHELL-1-LICENSE`. No other dataset audio is included. |

## Native changes

`vendor/qwentts/src/pipeline-tts.cpp` has two local changes. Reaching the frame
budget without EOS returns a generation failure, so our transactional WAV writer
cannot silently publish truncated speech. Single-request engines allocate Talker
KV only after the prompt is known, round prompt plus frame budget to 256-position
classes, and grow between requests up to the original 4096-position ceiling.
Growth releases old graph views and KV before allocating a replacement; batched
engines retain their original fixed allocation. Model precision is unchanged.
The modified file has a prominent notice. Apply the exact deltas in this order:
`third_party/patches/qwentts-generation-budget.patch`, then
`third_party/patches/qwentts-lazy-kv.patch`.

`vendor/qwentts/src/gguf-weights.h` and `weight-ctx.h` additionally bind
unconverted CPU tensors directly to immutable mapped GGUF storage. Converted
tensors retain owned buffers, non-CPU backends retain the copy path, and weight
buffers are destroyed before their GGUF mapping. No quantization or weight
values change. The exact delta is in
`third_party/patches/qwentts-mapped-cpu-weights.patch`. Our adapter uses the
pinned GGML internal multi-buffer helper to own mapped-buffer metadata and
converted-weight buffers together; no GGML source file is modified.

`native/CMakeLists.txt`, `native/bridge.cpp` and `build.rs` are new Valle code.
They build only the CPU pipeline, supply the pinned version identity and hide
upstream ABI structs behind a small compiled adapter. No upstream source is
fetched or executed during the build.

`third_party/sources.json` identifies source commits. `third_party/vendor-files.json`
lists every copied file and its current SHA-256; the changed pipeline also
records its original upstream hash. `scripts/check_vendor.py` fails on unexpected
files or modifications. Review upstream licensing and local deltas before
regenerating these records.

## Model weights

`models.json` pins [Serveurperso/Qwen3-TTS-GGUF](https://huggingface.co/Serveurperso/Qwen3-TTS-GGUF/tree/b7ee2e8c7459c3bea99da23e3d178125a7d1713c)
revision `b7ee2e8c7459c3bea99da23e3d178125a7d1713c`, file sizes and SHA-256 hashes.
These are community GGUF conversions, **not original official safetensors**.
The model repository declares Apache-2.0 and attributes the official
[Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)
and [12Hz tokenizer](https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz), both
Apache-2.0. Weights are downloaded separately and are not committed to Git.

## Rust dependencies

Direct crates were checked against the newest stable crates.io releases on
2026-10-09. Cargo.lock pins the resolved graph. `third_party/crates.json`
records dependencies, SPDX expressions and repositories across the supported
Linux, Windows and macOS targets. `scripts/check_licenses.py` selects permissive
options for dual-licensed crates and rejects missing or non-permissive choices.
Native MIT sources are separately verified, since Cargo metadata does not cover
vendored C++ code.
