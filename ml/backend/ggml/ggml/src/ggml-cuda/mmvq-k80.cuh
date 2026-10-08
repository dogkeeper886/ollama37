#include "common.cuh"

// Batch-1 q4_K mat-vec for Kepler (#563). Kepler has no dp4a and converts int to float at 1/6 of
// its FP32 FMA rate, so this path decodes each nibble to FP32 with bit operations instead.
bool ggml_cuda_should_use_mmvq_k80(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);

void ggml_cuda_mul_mat_vec_q_k80(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
