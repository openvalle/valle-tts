// Copyright 2026 Valle contributors. SPDX-License-Identifier: Apache-2.0
// Our small ABI hides upstream POD layout and pins it at compile time.
#include "qwen.h"
#include <new>

// &mut Qwen3 permits one synchronous request at a time. Always clear the
// borrowed callback before returning to Rust, including native failure paths.
struct compute_cancel_scope {
    qt_context * ctx;
    compute_cancel_scope(qt_context * ctx, qt_cancel_cb cancel, void * data) : ctx(ctx) {
        qt_set_compute_cancel(ctx, cancel, data);
    }
    ~compute_cancel_scope() { qt_set_compute_cancel(ctx, nullptr, nullptr); }
};

extern "C" {
void * vt_load(const char * talker, const char * codec) {
    qt_init_params p;
    qt_init_default_params(&p);
    p.talker_path = talker;
    p.codec_path = codec;
    p.use_fa = false;
    p.max_batch = 1;
    return qt_init(&p);
}
void vt_free(void * ctx) { qt_free(static_cast<qt_context *>(ctx)); }
const char * vt_error() { return qt_last_error(); }
const char * vt_model_type(void * ctx) { return qt_model_type(static_cast<qt_context *>(ctx)); }
const char * vt_version() { return qt_version(); }
void * vt_voice(void * ctx, const float * samples, int count, qt_cancel_cb cancel, void * cancel_data) {
    compute_cancel_scope scope(static_cast<qt_context *>(ctx), cancel, cancel_data);
    auto * ref = new (std::nothrow) qt_voice_ref{};
    if (!ref) return nullptr;
    if (qt_extract_voice_ref(static_cast<qt_context *>(ctx), samples, count, ref) != QT_STATUS_OK) {
        qt_voice_ref_free(ref);
        delete ref;
        return nullptr;
    }
    return ref;
}
void vt_voice_free(void * voice) {
    auto * ref = static_cast<qt_voice_ref *>(voice);
    qt_voice_ref_free(ref);
    delete ref;
}
struct vt_request {
    const char * text;
    const char * language;
    const char * transcript;
    const void * voice;
    int64_t seed;
    int max_tokens;
    float temperature;
    qt_audio_chunk_cb emit;
    qt_cancel_cb cancel;
    void * cancel_user_data;
    void * user_data;
};
int vt_synthesize(void * ctx, const vt_request * request) {
    compute_cancel_scope scope(static_cast<qt_context *>(ctx), request->cancel, request->cancel_user_data);
    qt_tts_params p;
    qt_tts_default_params(&p);
    p.text = request->text;
    p.lang = request->language;
    p.ref_text = request->transcript;
    p.seed = request->seed;
    p.max_new_tokens = request->max_tokens;
    p.temperature = request->temperature;
    p.on_chunk = request->emit;
    p.on_chunk_user_data = request->user_data;
    p.cancel = request->cancel;
    p.cancel_user_data = request->cancel_user_data;
    if (request->voice) {
        const auto * ref = static_cast<const qt_voice_ref *>(request->voice);
        p.ref_spk_emb = ref->ref_spk_emb;
        p.ref_spk_dim = ref->ref_spk_dim;
        if (request->transcript) {
            p.ref_codes = ref->ref_codes;
            p.ref_T = ref->ref_T;
        }
    }
    qt_audio out{};
    const int rc = qt_synthesize(static_cast<qt_context *>(ctx), &p, &out);
    qt_audio_free(&out);
    return rc;
}
}
