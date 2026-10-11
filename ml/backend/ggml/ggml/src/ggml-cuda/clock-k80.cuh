#include "common.cuh"

// Kepler's Autoboost clock in a multi-die decode (#587). A K80 die raises its clock only above ~58 % utilization
// and scales it down below ~50 %, and one die's share of a multi-die token is under that. So between its turns a
// die runs a one-thread filler kernel, which exits when the die's next turn starts. Call at the start of each
// graph compute.
void ggml_cuda_k80_clock_fill(ggml_backend_cuda_context & ctx);

void ggml_cuda_k80_clock_free(ggml_backend_cuda_context & ctx);
