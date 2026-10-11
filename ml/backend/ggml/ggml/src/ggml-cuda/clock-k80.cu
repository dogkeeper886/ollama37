#include "clock-k80.cuh"

static constexpr int64_t K80_FILL_MAX_PERIOD_US = 1000000; // slower turns than this are not a decode
static constexpr int     K80_FILL_POLLS_PER_MS  = 60;      // at 875 MHz

static ggml_backend_cuda_context * k80_fill_prev = nullptr; // the context of the last graph compute

// Spins on one thread until the host changes *flag (pinned host memory, read in place) or the polls run out.
static __global__ void k80_fill(const volatile int * flag, const int gen, const int polls) {
    float a = 1.0f;
    for (int i = 0; i < polls && *flag == gen && a > 0.0f; ++i) {
        for (int j = 0; j < 1000; ++j) {
            a = fmaf(a, 0.999999f, 1e-7f);
        }
    }
}

void ggml_cuda_k80_clock_fill(ggml_backend_cuda_context & ctx) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!(GGML_CUDA_CC_IS_NVIDIA(cc) && cc < 500)) {
        return;
    }

    if (ctx.k80_fill_flag == nullptr) {
        CUDA_CHECK(cudaHostAlloc(&ctx.k80_fill_flag, sizeof(int), cudaHostAllocMapped));
        *ctx.k80_fill_flag = 0;
        CUDA_CHECK(cudaEventCreateWithFlags(&ctx.k80_fill_event, cudaEventDisableTiming));
    }
    ++*ctx.k80_fill_flag; // this die's turn: its filler exits

    const int64_t t_us = ggml_time_us();
    ctx.k80_fill_period_us = t_us - ctx.k80_fill_t_us;
    ctx.k80_fill_t_us      = t_us;

    ggml_backend_cuda_context * prev = k80_fill_prev;
    k80_fill_prev = &ctx;
    if (prev == nullptr || prev == &ctx || prev->k80_fill_period_us > K80_FILL_MAX_PERIOD_US) {
        return;
    }

    // the previous die's turn is queued on its stream: fill from its end, for one period at most
    ggml_cuda_set_device(prev->device);
    cudaStream_t fill_stream = prev->stream(prev->device, 1);
    CUDA_CHECK(cudaEventRecord(prev->k80_fill_event, prev->stream()));
    CUDA_CHECK(cudaStreamWaitEvent(fill_stream, prev->k80_fill_event, 0));
    const int polls = (int) (prev->k80_fill_period_us*K80_FILL_POLLS_PER_MS/1000);
    k80_fill<<<1, 1, 0, fill_stream>>>(prev->k80_fill_flag, *prev->k80_fill_flag, polls);
    ggml_cuda_set_device(ctx.device);
}

void ggml_cuda_k80_clock_free(ggml_backend_cuda_context & ctx) {
    if (k80_fill_prev == &ctx) {
        k80_fill_prev = nullptr;
    }
    if (ctx.k80_fill_flag == nullptr) {
        return;
    }
    ++*ctx.k80_fill_flag;
    ggml_cuda_set_device(ctx.device);
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream(ctx.device, 1)));
    CUDA_CHECK(cudaEventDestroy(ctx.k80_fill_event));
    CUDA_CHECK(cudaFreeHost(ctx.k80_fill_flag));
}
