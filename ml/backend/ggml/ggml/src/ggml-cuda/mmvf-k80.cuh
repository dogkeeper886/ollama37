#include "common.cuh"

// Batch-1 F16 x F32 mat-vec with short rows for Kepler (#565): decode attention's KQ over the f16 key
// cache when flash attention is off. Each row is read once for every src1 head broadcast onto it.
bool ggml_cuda_should_use_mmvf_k80(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);

void ggml_cuda_mul_mat_vec_f_k80(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
