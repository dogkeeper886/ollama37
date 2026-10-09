#include "common.cuh"

// One Gated DeltaNet decode step per sequence (#571); see ggml_gated_delta_step.
bool ggml_cuda_gated_delta_step_supported(const ggml_tensor * dst);

void ggml_cuda_op_gated_delta_step(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
