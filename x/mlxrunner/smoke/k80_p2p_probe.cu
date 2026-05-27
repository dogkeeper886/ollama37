// K80 multi-die P2P probe — Phase D spike (#187).
//
// Before sinking more weeks into the runner and MoE kernel ports, answer the
// hardest unknown: can we shard a 19 GB model across 3 Tesla K80 cards
// (= 6 dies of 12 GB each) on this host? Specifically:
//
//   (1) How many K80 dies does CUDA actually see?
//   (2) Which pairs can access each other's memory (cudaDeviceCanAccessPeer)?
//   (3) When P2P is enabled, what's the cudaMemcpy bandwidth across dies?
//   (4) Can we allocate and use ~1 GB independently on each die?
//   (5) Does a tiny compute kernel produce correct results on every die?
//
// Pure CUDA runtime — no MLX dependency. Even if MLX's Device abstraction
// turns out to be single-device-only (it likely is), the hardware capability
// answer comes first. If P2P is fast enough across K80 dies, the path forward
// is "extend MLX's Device API"; if P2P is impossible or slow, the path
// changes to either host-RAM expert paging or "stop here, this won't work."
//
// Build via x/mlxrunner/CMakeLists.txt as a third exe target.
// Run inside the runtime container with --gpus all (no CUDA_VISIBLE_DEVICES
// restriction, unlike qwen_smoke / qwen_load which use device 0 only).

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

// Trivial compute: out[i] = in[i] * 2 + 1. Lets us verify per-die compute
// without needing cuBLAS or any other library.
__global__ void scale_add_kernel(const float* in, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = in[i] * 2.0f + 1.0f;
  }
}

struct DeviceInfo {
  int id;
  cudaDeviceProp prop;
};

int main() {
  int num_devices = 0;
  CHECK_CUDA(cudaGetDeviceCount(&num_devices));
  std::printf("k80_p2p: %d CUDA device(s) visible\n", num_devices);

  std::vector<DeviceInfo> devs;
  devs.reserve(num_devices);
  for (int d = 0; d < num_devices; d++) {
    DeviceInfo di{d, {}};
    CHECK_CUDA(cudaGetDeviceProperties(&di.prop, d));
    std::printf("  device %d: %s  cc=%d.%d  totalMem=%.1f MiB\n",
                d, di.prop.name, di.prop.major, di.prop.minor,
                di.prop.totalGlobalMem / 1024.0 / 1024.0);
    devs.push_back(di);
  }

  if (num_devices < 2) {
    std::printf("k80_p2p: only %d device(s) — multi-die test needs >=2\n",
                num_devices);
    std::printf("k80_p2p: probe OK (single-die mode)\n");
    return 0;
  }

  // ===== (2) P2P access matrix =====
  std::printf("\nk80_p2p: P2P access matrix (cudaDeviceCanAccessPeer):\n      ");
  for (int j = 0; j < num_devices; j++) std::printf(" d%d", j);
  std::printf("\n");
  std::vector<std::vector<int>> peer_ok(num_devices, std::vector<int>(num_devices, 0));
  for (int i = 0; i < num_devices; i++) {
    std::printf("  d%d  ", i);
    for (int j = 0; j < num_devices; j++) {
      if (i == j) {
        std::printf("  -");
        continue;
      }
      int can = 0;
      CHECK_CUDA(cudaDeviceCanAccessPeer(&can, i, j));
      peer_ok[i][j] = can;
      std::printf("  %d", can);
    }
    std::printf("\n");
  }

  // Count peer pairs to decide whether we can do any meaningful sharding.
  int peer_pairs = 0;
  for (int i = 0; i < num_devices; i++)
    for (int j = 0; j < num_devices; j++)
      if (i != j && peer_ok[i][j]) peer_pairs++;
  std::printf("k80_p2p: %d directional peer pair(s) of %d possible\n",
              peer_pairs, num_devices * (num_devices - 1));

  // ===== (4) Allocate 1 GB on each die (independent VRAM check) =====
  constexpr size_t kBytes = 256u * 1024 * 1024;  // 256 MiB per die (keep
                                                  // session-friendly; spec
                                                  // says 12 GiB total per die)
  std::vector<float*> bufs(num_devices, nullptr);
  std::printf("\nk80_p2p: allocating %zu MiB on each die...\n",
              kBytes / 1024 / 1024);
  for (int d = 0; d < num_devices; d++) {
    CHECK_CUDA(cudaSetDevice(d));
    cudaError_t e = cudaMalloc(&bufs[d], kBytes);
    if (e != cudaSuccess) {
      std::fprintf(stderr, "  die %d alloc FAILED: %s\n",
                   d, cudaGetErrorString(e));
      std::exit(3);
    }
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    std::printf("  d%d: %.0f MiB free of %.0f MiB total after alloc\n", d,
                free_b / 1024.0 / 1024.0, total_b / 1024.0 / 1024.0);
  }

  // ===== (5) Tiny compute on each die, validate result =====
  const int N = 1024 * 1024;  // 4 MiB worth of floats
  std::vector<float> host_in(N, 0.5f);
  std::vector<float> host_out(N);
  std::printf("\nk80_p2p: per-die compute check (scale_add on %d floats)...\n", N);
  for (int d = 0; d < num_devices; d++) {
    CHECK_CUDA(cudaSetDevice(d));
    CHECK_CUDA(cudaMemcpy(bufs[d], host_in.data(), N * sizeof(float),
                          cudaMemcpyHostToDevice));
    int block = 256;
    int grid = (N + block - 1) / block;
    scale_add_kernel<<<grid, block>>>(bufs[d], bufs[d] + N / 2, N / 2);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(host_out.data(), bufs[d] + N / 2,
                          (N / 2) * sizeof(float), cudaMemcpyDeviceToHost));
    bool ok = (host_out[0] == 2.0f) && (host_out[N / 2 - 1] == 2.0f);
    std::printf("  d%d: out[0]=%.1f out[N/2-1]=%.1f -> %s\n", d,
                host_out[0], host_out[N / 2 - 1], ok ? "OK" : "WRONG");
    if (!ok) std::exit(4);
  }

  // ===== (3) Bandwidth: cudaMemcpy across dies =====
  // Try every directional pair where peer is allowed. For pairs that aren't
  // peer-allowed, also measure cudaMemcpyDeviceToDevice (which falls back
  // to host-staging). That tells us the worst-case bandwidth we'd be stuck
  // with for sharding.
  std::printf("\nk80_p2p: cross-die bandwidth (cudaMemcpy %zu MiB):\n",
              kBytes / 1024 / 1024);
  for (int src = 0; src < num_devices; src++) {
    for (int dst = 0; dst < num_devices; dst++) {
      if (src == dst) continue;
      bool p2p = peer_ok[src][dst];

      if (p2p) {
        CHECK_CUDA(cudaSetDevice(src));
        cudaError_t e = cudaDeviceEnablePeerAccess(dst, 0);
        if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) {
          std::fprintf(stderr, "  d%d->d%d enable FAILED: %s\n", src, dst,
                       cudaGetErrorString(e));
          continue;
        }
      }
      CHECK_CUDA(cudaSetDevice(src));
      // Warm up
      CHECK_CUDA(cudaMemcpyPeer(bufs[dst], dst, bufs[src], src, kBytes));
      CHECK_CUDA(cudaDeviceSynchronize());

      // Measure
      constexpr int kIter = 4;
      auto t0 = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < kIter; i++) {
        CHECK_CUDA(cudaMemcpyPeer(bufs[dst], dst, bufs[src], src, kBytes));
      }
      CHECK_CUDA(cudaDeviceSynchronize());
      auto t1 = std::chrono::high_resolution_clock::now();
      double sec = std::chrono::duration<double>(t1 - t0).count();
      double gbps = (kBytes * double(kIter)) / sec / 1e9;
      std::printf("  d%d -> d%d: %s   %.2f GB/s\n", src, dst,
                  p2p ? "[P2P ]" : "[stage]", gbps);
    }
  }

  // Cleanup
  for (int d = 0; d < num_devices; d++) {
    cudaSetDevice(d);
    cudaFree(bufs[d]);
  }

  std::printf("\nk80_p2p: probe OK\n");
  return 0;
}
