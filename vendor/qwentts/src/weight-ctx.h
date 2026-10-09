#pragma once
// weight-ctx.h: format-independent weight loading context for ggml backends
//
// Manages a ggml_context for weight tensors + their backend buffer.
// Used by gguf-weights.h for all model loaders.
//
// Usage:
//   WeightCtx wctx;
//   wctx_init(&wctx, n_tensors);
//   ggml_tensor * w = <loader>_load_tensor(&wctx, source, "name");
//   wctx_alloc(&wctx, backend);

// Modified by Valle contributors: CPU tensors may borrow immutable GGUF mmap
// storage. Converted tensors retain owned storage. The owning GGUFModel must
// outlive all weight buffers, as it already does in the pipeline teardown.

#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "qt-error.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

struct WeightCtx {
    struct ggml_context * ctx;
    ggml_backend_buffer_t buffer;
    const void * mapped_base;
    size_t mapped_size;

    struct PendingCopy {
        struct ggml_tensor * tensor;
        const void *         src;
        size_t               nbytes;
        size_t               offset;  // byte offset into dst tensor (0 for regular loads)
        bool                 mapped = false;
    };

    std::vector<PendingCopy> pending;

    // Staging buffers for type-converted data, kept alive until wctx_alloc.
    // unique_ptr keeps the data address stable even when the outer vector grows,
    // so src pointers stored in pending stay valid across staging.push_back().
    std::vector<std::unique_ptr<float[]>> staging;
};

static void wctx_init(WeightCtx * wctx, int n_tensors) {
    size_t                  ctx_size = (size_t) n_tensors * ggml_tensor_overhead() + 1024;
    struct ggml_init_params params   = {
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/true,
    };
    wctx->ctx    = ggml_init(params);
    wctx->buffer = NULL;
    wctx->mapped_base = NULL;
    wctx->mapped_size = 0;
    wctx->pending.clear();
    wctx->pending.reserve(n_tensors);
}

static bool wctx_alloc(WeightCtx * wctx, ggml_backend_t backend) {
    ggml_backend_buffer_t mapped = NULL;
    size_t mapped_bytes = 0;
    const size_t alignment = ggml_backend_buft_get_alignment(ggml_backend_cpu_buffer_type());
    if (ggml_backend_is_cpu(backend) && wctx->mapped_base &&
        (uintptr_t) wctx->mapped_base % alignment == 0) {
        // This buffer owns metadata only, never the read-only mapping itself.
        mapped = ggml_backend_cpu_buffer_from_ptr(const_cast<void *>(wctx->mapped_base), wctx->mapped_size);
        if (!mapped) {
            return false;
        }
        ggml_backend_buffer_set_usage(mapped, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (auto & pc : wctx->pending) {
            const uintptr_t base = (uintptr_t) wctx->mapped_base;
            const uintptr_t src = (uintptr_t) pc.src;
            if (!pc.mapped || pc.offset != 0 || pc.nbytes != ggml_nbytes(pc.tensor) ||
                src < base || src - base > wctx->mapped_size ||
                pc.nbytes > wctx->mapped_size - (src - base)) {
                continue;
            }
            if (ggml_backend_tensor_alloc(mapped, pc.tensor, const_cast<void *>(pc.src)) != GGML_STATUS_SUCCESS) {
                ggml_backend_buffer_free(mapped);
                return false;
            }
            mapped_bytes += pc.nbytes;
        }
    }

    bool needs_owned = false;
    for (auto * tensor = ggml_get_first_tensor(wctx->ctx); tensor; tensor = ggml_get_next_tensor(wctx->ctx, tensor)) {
        needs_owned |= tensor->data == NULL && tensor->view_src == NULL;
    }
    // The allocator skips tensors that are already bound to the mapping.
    ggml_backend_buffer_t owned = needs_owned ? ggml_backend_alloc_ctx_tensors(wctx->ctx, backend) : NULL;
    if (needs_owned && !owned) {
        ggml_backend_buffer_free(mapped);
        qt_log(QT_LOG_ERROR, "[WeightCtx] FATAL: failed to allocate converted weights");
        return false;
    }
    if (mapped && owned) {
        ggml_backend_buffer_t buffers[] = { mapped, owned };
        wctx->buffer = ggml_backend_multi_buffer_alloc_buffer(buffers, 2);
    } else {
        wctx->buffer = mapped ? mapped : owned;
    }
    if (!wctx->buffer) {
        return false;
    }
    // Mark as weight buffer so ggml_backend_sched assigns ops to the correct
    // backend based on weight location (avoids fallback through expansion).
    ggml_backend_buffer_set_usage(wctx->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    size_t total = 0;
    for (auto & pc : wctx->pending) {
        if (pc.tensor->buffer != mapped) {
            ggml_backend_tensor_set(pc.tensor, pc.src, pc.offset, pc.nbytes);
        }
        total += pc.nbytes;
    }
    qt_log(QT_LOG_INFO, "[WeightCtx] Loaded %zu tensors, %.1f MB (%.1f MB mapped, %.1f MB copied)", wctx->pending.size(),
           (float) total / (1024 * 1024), (float) mapped_bytes / (1024 * 1024),
           (float) (total - mapped_bytes) / (1024 * 1024));
    wctx->pending.clear();
    wctx->staging.clear();
    return true;
}

static void wctx_free(WeightCtx * wctx) {
    if (wctx->buffer) {
        ggml_backend_buffer_free(wctx->buffer);
    }
    if (wctx->ctx) {
        ggml_free(wctx->ctx);
    }
    wctx->buffer = NULL;
    wctx->ctx    = NULL;
}
