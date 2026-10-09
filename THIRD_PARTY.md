# Third-party provenance and licenses

This document records external implementation references, copied or modified
code, test fixtures, model weights and dependency licenses. Valle's original
code is Apache-2.0, as declared in the repository [LICENSE](LICENSE).
Third-party material retains its upstream license and notices.

This repository has a new Rust API, model dispatch, CLI, audio frontend, WAV
writer and safe native adapter. **The Qwen inference core is copied and linked
C++ source, not a new pure Rust neural network implementation.** Its existing
MIT license and notices are preserved. This distinction is intentional.

| Source | Pinned revision | License | Exact use |
|---|---|---|---|
| [ServeurpersoCom/qwentts.cpp](https://github.com/ServeurpersoCom/qwentts.cpp/tree/51512f129a7419567f4b8abfb06801451789b8f1) | `51512f129a7419567f4b8abfb06801451789b8f1` | MIT | `src/` copied to `vendor/qwentts/src/`; the three pipeline translation units are compiled and statically linked. No CLI, HTTP server, converter or third-party HTTP/JSON library is built. Upstream LICENSE retained. |
| [ServeurpersoCom/ggml](https://github.com/ServeurpersoCom/ggml/tree/2eddaf94a8cdbbf1c30017d28313cd661a9da543) | `2eddaf94a8cdbbf1c30017d28313cd661a9da543` | MIT | The upstream's exact ggml submodule, copied under `vendor/ggml/` without Git metadata, examples, tests, CI, docs and non-CPU backend source directories. Only CPU/core code is compiled; GPU/third-party acceleration paths are disabled. LICENSE retained. |
| [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) | Model revisions below | Apache-2.0 | Official architecture/model documentation reference; no Python source copied into our Rust code. Full license in `third_party/QwenLM__Qwen3-TTS-LICENSE`. |
| [alan890104/qwen3-asr-rs](https://github.com/alan890104/qwen3-asr-rs/tree/c5ef09646af6278d2ba8b8ceaf543ffb32d1a5dc) | `c5ef09646af6278d2ba8b8ceaf543ffb32d1a5dc` | MIT | Copied English `tests/fixtures/sample1.wav` and `sample1.txt`; no inference code from this project is used. Original source hashes are in `tests/fixtures/provenance.json`; only sample1 is included here. |
| [AISHELL-1](https://www.openslr.org/33/) | Mirror `2724409d538167445e43ebf846990319f12a1cbf` | Apache-2.0 | One Chinese reference WAV and its original transcript. Source path, hash and pinned mirror are in `tests/fixtures/zh-provenance.json`; license in `third_party/AISHELL-1-LICENSE`. No other dataset audio is included. |

Full upstream licenses and their copyright notices are preserved verbatim:

| Source | Preserved license |
|---|---|
| qwentts.cpp | [MIT notice](vendor/qwentts/LICENSE) |
| ggml | [MIT notice](vendor/ggml/LICENSE) |
| Qwen3-TTS | [Apache-2.0 notice](third_party/QwenLM__Qwen3-TTS-LICENSE) |
| qwen3-asr-rs fixture | [MIT notice](third_party/alan890104__qwen3-asr-rs-LICENSE) |
| AISHELL-1 fixture | [Apache-2.0 notice](third_party/AISHELL-1-LICENSE) |

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

## Test fixtures

[tests/fixtures/provenance.json](tests/fixtures/provenance.json) records the
external source revision, sizes and SHA-256 hashes of the English WAV and
transcript. [tests/fixtures/zh-provenance.json](tests/fixtures/zh-provenance.json)
records the AISHELL-1 source, mirror revision, original dataset path, sample ID,
WAV hash and Chinese transcript. The corresponding MIT and Apache-2.0 notices
are retained in `third_party/`.

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
records external dependencies, SPDX expressions and repositories across the
supported Linux, Windows and macOS targets. Workspace packages are excluded
from this third-party report. `scripts/check_licenses.py` selects permissive
options for dual-licensed crates and rejects missing or non-permissive choices.
Native MIT sources are separately verified, since Cargo metadata does not cover
vendored C++ code.

## Distribution and updates

`Cargo.toml` includes this document, `third_party/`, the complete pinned native
sources and their MIT licenses, and fixture provenance in the published source
archive. Model weights and generated test output are downloaded or produced
separately and are excluded from that archive.

When updating external code or dependencies, verify the actual license and
pinned revision, retain upstream notices, document local modifications, and
regenerate the source/dependency reports. Run `scripts/check_vendor.py` and
`scripts/check_licenses.py` before merging those changes.
