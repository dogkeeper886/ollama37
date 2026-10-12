#include "common.cuh"

// MUL_MAT_ID at batch > 1 (prefill) for Kepler (#583). Without dp4a, ggml's path syncs with the host for
// the routing and runs each expert as its own dequantize + SGEMM over a handful of tokens. This path routes
// on the device and runs every expert in one FP32 GEMM launch that decodes q4_K/q6_K tiles into shared memory.
bool ggml_cuda_should_use_mmid_k80(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst, int cc);

void ggml_cuda_mul_mat_id_k80(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst);
