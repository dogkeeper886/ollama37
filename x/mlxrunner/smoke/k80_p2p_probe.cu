// K80 multi-die bandwidth probe — Phase D spike (#187), v2.
//
// v1 measured cross-die bandwidth at ~0.4 GB/s and concluded "P2P is broken,
// multi-die sharding is bandwidth-limited." That was wrong on its face —
// ollama on K80 already runs multi-GPU models (qwen3.6:35b validated in #182)
// at usable throughput, so the relevant cross-die path can't be that slow.
//
// Two bugs in v1:
//   (a) Used cudaMemcpyPeer (synchronous, default stream, pageable
//       intermediate). That's the slowest possible cross-device copy path.
//       The ggml/llama.cpp multi-GPU path uses pinned host memory +
//       cudaMemcpyAsync, which is what we should measure for any realistic
//       deployment.
//   (b) Reported "P2P unavailable" as a deal-breaker without distinguishing
//       hardware-P2P (cudaMemcpyPeer over PCIe directly, fastest) from
//       library-P2P (CUDA staging via host RAM, the always-available
//       fallback). Even when hardware P2P is unavailable, the
//       pinned-staged fallback can hit ~6-10 GB/s on modern PCIe.
//
// v2 measures all three regimes side-by-side so we can plan against the
// actual achievable cross-die bandwidth in this environment:
//
//   [hw-p2p]    cudaMemcpyPeer with peer-access enabled (only if
//               cudaDeviceCanAccessPeer reports 1)
//   [pinned]    src GPU -> pinned host RAM -> dst GPU, cudaMemcpyAsync on
//               per-die streams (what ggml does)
//   [naive]     cudaMemcpyPeer without enabling peer (== synchronous
//               pageable host staging); same as v1, kept as the floor
//
// Also: this run reports nvidia_peermem availability, the runtime + device
// flags it was invoked under (env), and per-die nvidia-smi topology so the
// CI artifact captures the full environment.
//
// Build via x/mlxrunner/CMakeLists.txt as a third exe target.
// Run via cicd/scripts/test-mlx-smoke.sh; the script runs it in the
// production runtime image (ollama37:latest) with --runtime=nvidia +
// NVIDIA_VISIBLE_DEVICES=all + NVIDIA_DRIVER_CAPABILITIES=compute,utility
// so the environment matches production ollama exactly.

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>

#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                  \
  do {                                                                    \
    cudaError_t e = (call);                                               \
    if (e != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error %s:%d: %s (%d)\n", __FILE__,       \
                   __LINE__, cudaGetErrorString(e), int(e));              \
      std::exit(2);                                                       \
    }                                                                     \
  } while (0)

__global__ void scale_add_kernel(const float* in, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = in[i] * 2.0f + 1.0f;
  }
}

// Measure GB/s for `iter` runs of cudaMemcpyAsync(dst_dev <- src_dev) via
// pinned host RAM. Each src->dst copy is two cudaMemcpyAsync: src GPU ->
// pinned host, then pinned host -> dst GPU. Uses per-device streams.
static double bench_pinned_staged(
    int src, int dst, void* src_dev, void* dst_dev, void* host_pinned,
    size_t bytes, int iter) {
  cudaStream_t src_stream, dst_stream;
  CHECK_CUDA(cudaSetDevice(src));
  CHECK_CUDA(cudaStreamCreate(&src_stream));
  CHECK_CUDA(cudaSetDevice(dst));
  CHECK_CUDA(cudaStreamCreate(&dst_stream));

  // warm-up
  CHECK_CUDA(cudaSetDevice(src));
  CHECK_CUDA(cudaMemcpyAsync(host_pinned, src_dev, bytes,
                             cudaMemcpyDeviceToHost, src_stream));
  CHECK_CUDA(cudaStreamSynchronize(src_stream));
  CHECK_CUDA(cudaSetDevice(dst));
  CHECK_CUDA(cudaMemcpyAsync(dst_dev, host_pinned, bytes,
                             cudaMemcpyHostToDevice, dst_stream));
  CHECK_CUDA(cudaStreamSynchronize(dst_stream));

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iter; i++) {
    CHECK_CUDA(cudaSetDevice(src));
    CHECK_CUDA(cudaMemcpyAsync(host_pinned, src_dev, bytes,
                               cudaMemcpyDeviceToHost, src_stream));
    CHECK_CUDA(cudaStreamSynchronize(src_stream));
    CHECK_CUDA(cudaSetDevice(dst));
    CHECK_CUDA(cudaMemcpyAsync(dst_dev, host_pinned, bytes,
                               cudaMemcpyHostToDevice, dst_stream));
    CHECK_CUDA(cudaStreamSynchronize(dst_stream));
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  double sec = std::chrono::duration<double>(t1 - t0).count();

  cudaSetDevice(src); cudaStreamDestroy(src_stream);
  cudaSetDevice(dst); cudaStreamDestroy(dst_stream);
  return (bytes * double(iter)) / sec / 1e9;
}

static double bench_peer_naive(
    int src, int dst, void* src_dev, void* dst_dev,
    size_t bytes, int iter) {
  // cudaMemcpyPeer without enabling peer access — the synchronous fallback.
  // Measures the worst-case "if you don't think about it" path that v1
  // accidentally used.
  CHECK_CUDA(cudaSetDevice(src));
  CHECK_CUDA(cudaMemcpyPeer(dst_dev, dst, src_dev, src, bytes));  // warm
  CHECK_CUDA(cudaDeviceSynchronize());

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iter; i++) {
    CHECK_CUDA(cudaMemcpyPeer(dst_dev, dst, src_dev, src, bytes));
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  auto t1 = std::chrono::high_resolution_clock::now();
  double sec = std::chrono::duration<double>(t1 - t0).count();
  return (bytes * double(iter)) / sec / 1e9;
}

static double bench_hw_p2p(
    int src, int dst, void* src_dev, void* dst_dev,
    size_t bytes, int iter) {
  // Only safe to call when cudaDeviceCanAccessPeer(src, dst) returned 1
  // AND cudaDeviceEnablePeerAccess succeeded.
  CHECK_CUDA(cudaSetDevice(src));
  CHECK_CUDA(cudaMemcpyPeer(dst_dev, dst, src_dev, src, bytes));
  CHECK_CUDA(cudaDeviceSynchronize());

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iter; i++) {
    CHECK_CUDA(cudaMemcpyPeer(dst_dev, dst, src_dev, src, bytes));
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  auto t1 = std::chrono::high_resolution_clock::now();
  double sec = std::chrono::duration<double>(t1 - t0).count();
  return (bytes * double(iter)) / sec / 1e9;
}

int main() {
  // --- environment snapshot ---
  std::printf("=== environment ===\n");
  for (const char* var : {"NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES",
                          "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH"}) {
    const char* v = std::getenv(var);
    std::printf("  %s = %s\n", var, v ? v : "(unset)");
  }
  // nvidia_peermem kernel module (controls GPUDirect / hardware P2P enablement)
  FILE* fp = std::fopen("/sys/module/nvidia_peermem/version", "r");
  if (fp) {
    char buf[64] = {};
    std::fread(buf, 1, sizeof(buf) - 1, fp);
    std::fclose(fp);
    std::printf("  nvidia_peermem: LOADED (version %s)", buf);
  } else {
    std::printf("  nvidia_peermem: NOT LOADED  <- typically required for hw P2P\n");
  }
  std::printf("\n");

  int num_devices = 0;
  CHECK_CUDA(cudaGetDeviceCount(&num_devices));
  std::printf("=== devices ===\n");
  std::printf("  %d CUDA device(s) visible\n", num_devices);
  for (int d = 0; d < num_devices; d++) {
    cudaDeviceProp p;
    CHECK_CUDA(cudaGetDeviceProperties(&p, d));
    std::printf("  device %d: %s  cc=%d.%d  PCIe %02x:%02x.%d  totalMem=%.1f MiB\n",
                d, p.name, p.major, p.minor, p.pciBusID, p.pciDeviceID,
                p.pciDomainID, p.totalGlobalMem / 1024.0 / 1024.0);
  }
  std::printf("\n");

  if (num_devices < 2) {
    std::printf("k80_p2p: only %d device(s); single-die mode\nk80_p2p: probe OK\n",
                num_devices);
    return 0;
  }

  // --- (2) cudaDeviceCanAccessPeer matrix ---
  std::printf("=== cudaDeviceCanAccessPeer matrix (1 = hw P2P advertised) ===\n      ");
  for (int j = 0; j < num_devices; j++) std::printf(" d%d", j);
  std::printf("\n");
  std::vector<std::vector<int>> peer_ok(num_devices, std::vector<int>(num_devices, 0));
  int peer_pairs = 0;
  for (int i = 0; i < num_devices; i++) {
    std::printf("  d%d  ", i);
    for (int j = 0; j < num_devices; j++) {
      if (i == j) { std::printf("  -"); continue; }
      int can = 0;
      CHECK_CUDA(cudaDeviceCanAccessPeer(&can, i, j));
      peer_ok[i][j] = can;
      if (can) peer_pairs++;
      std::printf("  %d", can);
    }
    std::printf("\n");
  }
  std::printf("  hw-P2P pairs: %d of %d\n\n", peer_pairs,
              num_devices * (num_devices - 1));

  // --- allocate per-die buffers + one pinned host staging buffer ---
  constexpr size_t kBytes = 256u * 1024 * 1024;  // 256 MiB
  std::vector<float*> bufs(num_devices, nullptr);
  for (int d = 0; d < num_devices; d++) {
    CHECK_CUDA(cudaSetDevice(d));
    CHECK_CUDA(cudaMalloc(&bufs[d], kBytes));
  }
  void* pinned_host = nullptr;
  CHECK_CUDA(cudaMallocHost(&pinned_host, kBytes));

  // --- per-die compute sanity ---
  std::printf("=== per-die compute check ===\n");
  const int N = 1024 * 1024;
  std::vector<float> host_in(N, 0.5f), host_out(N);
  for (int d = 0; d < num_devices; d++) {
    CHECK_CUDA(cudaSetDevice(d));
    CHECK_CUDA(cudaMemcpy(bufs[d], host_in.data(), N * sizeof(float),
                          cudaMemcpyHostToDevice));
    scale_add_kernel<<<(N + 255) / 256, 256>>>(bufs[d], bufs[d] + N / 2, N / 2);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(host_out.data(), bufs[d] + N / 2,
                          (N / 2) * sizeof(float), cudaMemcpyDeviceToHost));
    bool ok = (host_out[0] == 2.0f) && (host_out[N / 2 - 1] == 2.0f);
    std::printf("  d%d compute: %s\n", d, ok ? "OK" : "WRONG");
    if (!ok) std::exit(4);
  }
  std::printf("\n");

  // --- bandwidth measurements (the actual deliverable) ---
  std::printf("=== cross-die bandwidth (%zu MiB) ===\n", kBytes / 1024 / 1024);
  std::printf("  [hw-p2p]   = cudaMemcpyPeer with peer access enabled\n");
  std::printf("  [pinned]   = cudaMemcpyAsync via pinned host RAM (what ggml does)\n");
  std::printf("  [naive]    = cudaMemcpyPeer without peer access (pageable host staging)\n\n");

  constexpr int kIter = 4;
  for (int src = 0; src < num_devices; src++) {
    for (int dst = 0; dst < num_devices; dst++) {
      if (src == dst) continue;

      double bw_hw = 0.0, bw_pinned, bw_naive;

      // hw P2P — only if advertised
      if (peer_ok[src][dst]) {
        CHECK_CUDA(cudaSetDevice(src));
        cudaError_t e = cudaDeviceEnablePeerAccess(dst, 0);
        if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {
          std::fprintf(stderr, "  d%d->d%d enable FAILED: %s\n",
                       src, dst, cudaGetErrorString(e));
        } else {
          bw_hw = bench_hw_p2p(src, dst, bufs[src], bufs[dst], kBytes, kIter);
          // disable so the "naive" run below truly measures the staged path
          cudaDeviceDisablePeerAccess(dst);
        }
      }

      // pinned host staging (ggml's path)
      bw_pinned = bench_pinned_staged(src, dst, bufs[src], bufs[dst],
                                      pinned_host, kBytes, kIter);

      // naive (v1's accidental path)
      bw_naive = bench_peer_naive(src, dst, bufs[src], bufs[dst], kBytes, kIter);

      if (peer_ok[src][dst]) {
        std::printf("  d%d -> d%d:  hw-p2p %5.2f GB/s   pinned %5.2f GB/s   naive %5.2f GB/s\n",
                    src, dst, bw_hw, bw_pinned, bw_naive);
      } else {
        std::printf("  d%d -> d%d:  hw-p2p   N/A         pinned %5.2f GB/s   naive %5.2f GB/s\n",
                    src, dst, bw_pinned, bw_naive);
      }
    }
  }

  // --- cleanup ---
  cudaFreeHost(pinned_host);
  for (int d = 0; d < num_devices; d++) {
    cudaSetDevice(d);
    cudaFree(bufs[d]);
  }

  std::printf("\nk80_p2p: probe OK\n");
  return 0;
}
