// K80 multi-die bandwidth probe — Phase D spike (#187), v3.
//
// v2.1 reported per-die PCIe at 0.4 GB/s. That turned out to be a
// measurement artifact: the K80s were sitting in P8 (deep idle) when the
// probe ran. In P8:
//   - memory clock 324 MHz (vs boost 2505 MHz — 8x lower bandwidth)
//   - PCIe link Gen1 (vs max Gen3 — 4x lower)
//   - persistence_mode=Disabled (cold driver, slow upshift)
// Combined ~32x below spec. PCIe Gen3 spec ~14 GB/s / 32 = ~0.44 GB/s.
// Matches exactly what we measured.
//
// v3 wakes the GPUs up before measuring:
//
//   (a) [outside the probe, in test-mlx-smoke.sh] persistence mode +
//       application clocks lock are set on the host before docker run,
//       so the driver stays loaded across container invocations and the
//       GPU is pinned to boost clocks for the test window.
//
//   (b) [inside the probe] a compute warm-up phase launches a sustained
//       arithmetic kernel on each die for ~2 seconds, which forces the
//       SM cluster to P0 and pulls the PCIe link up to Gen3 alongside.
//
//   (c) [inside the probe] the P-state / clocks / PCIe link gen are
//       printed at three points (start, after warm-up, after measurement)
//       via nvmlDeviceGet* — so a future surprise like "stuck in P8 even
//       after warm-up" surfaces in the CI artifact instead of as
//       under-spec bandwidth.

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <nvml.h>

#define CHECK_CUDA(call)                                                  \
  do {                                                                    \
    cudaError_t e = (call);                                               \
    if (e != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error %s:%d: %s (%d)\n", __FILE__,       \
                   __LINE__, cudaGetErrorString(e), int(e));              \
      std::exit(2);                                                       \
    }                                                                     \
  } while (0)

#define CHECK_NVML(call)                                                  \
  do {                                                                    \
    nvmlReturn_t r = (call);                                              \
    if (r != NVML_SUCCESS) {                                              \
      std::fprintf(stderr, "NVML error %s:%d: %s\n", __FILE__, __LINE__,  \
                   nvmlErrorString(r));                                   \
    }                                                                     \
  } while (0)

// Compute-bound warm-up: each thread does a big arithmetic loop. Sustained
// launches put the SM cluster into P0 (and the driver typically upshifts
// PCIe alongside, though we'll verify via nvml).
__global__ void warmup_compute(float* out, int iters) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float v = 0.001f * static_cast<float>(i & 1023);
  for (int k = 0; k < iters; k++) {
    v = v * 1.0001f + 0.0001f;
  }
  out[i] = v;
}

__global__ void scale_add_kernel(const float* in, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i] * 2.0f + 1.0f;
}

// Print the current P-state / clocks / PCIe link gen for every GPU via NVML.
static void dump_gpu_state(const char* tag, int num_devices) {
  std::printf("--- gpu state [%s] ---\n", tag);
  for (int d = 0; d < num_devices; d++) {
    nvmlDevice_t h;
    if (nvmlDeviceGetHandleByIndex(d, &h) != NVML_SUCCESS) continue;

    nvmlPstates_t ps;
    unsigned int sm_clock = 0, mem_clock = 0;
    unsigned int link_gen = 0, link_width = 0;
    nvmlUtilization_t util{};

    nvmlDeviceGetPerformanceState(h, &ps);
    nvmlDeviceGetClockInfo(h, NVML_CLOCK_SM, &sm_clock);
    nvmlDeviceGetClockInfo(h, NVML_CLOCK_MEM, &mem_clock);
    nvmlDeviceGetCurrPcieLinkGeneration(h, &link_gen);
    nvmlDeviceGetCurrPcieLinkWidth(h, &link_width);
    nvmlDeviceGetUtilizationRates(h, &util);

    std::printf("  d%d: P%-2d  sm=%4u MHz  mem=%4u MHz  PCIe Gen%u x%u  util=%u%%\n",
                d, int(ps), sm_clock, mem_clock, link_gen, link_width, util.gpu);
  }
}

static double bench_pinned_staged(
    int src, int dst, void* src_dev, void* dst_dev, void* host_pinned,
    size_t bytes, int iter) {
  cudaStream_t src_stream, dst_stream;
  CHECK_CUDA(cudaSetDevice(src));
  CHECK_CUDA(cudaStreamCreate(&src_stream));
  CHECK_CUDA(cudaSetDevice(dst));
  CHECK_CUDA(cudaStreamCreate(&dst_stream));

  // warm-up transfer
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

static double bench_hw_p2p(
    int src, int dst, void* src_dev, void* dst_dev,
    size_t bytes, int iter) {
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
  FILE* fp = std::fopen("/sys/module/nvidia_peermem/version", "r");
  if (fp) { std::printf("  nvidia_peermem: LOADED\n"); std::fclose(fp); }
  else { std::printf("  nvidia_peermem: NOT LOADED\n"); }
  std::printf("\n");

  // --- NVML init (for clock/P-state readout) ---
  CHECK_NVML(nvmlInit_v2());

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
    nvmlShutdown(); return 0;
  }

  // --- (0) state BEFORE we do anything ---
  dump_gpu_state("at start, cold", num_devices);
  std::printf("\n");

  // --- (1) per-die compute warm-up to pull each GPU out of P8 ---
  // Allocate a small dummy output per die; launch a long arithmetic kernel
  // that the driver can't optimize out. Run for ~2 s per die — long enough
  // that the SM cluster transitions to P0 and the PCIe link upshifts.
  std::printf("=== warm-up (2s/die of sustained compute to wake from P8) ===\n");
  constexpr int kWarmThreads = 65536;
  constexpr int kWarmIters   = 200000;     // tune for ~2 s on K80 sm_37
  std::vector<float*> warm_out(num_devices, nullptr);
  for (int d = 0; d < num_devices; d++) {
    CHECK_CUDA(cudaSetDevice(d));
    CHECK_CUDA(cudaMalloc(&warm_out[d], kWarmThreads * sizeof(float)));
    auto t0 = std::chrono::high_resolution_clock::now();
    // Launch repeatedly until ~2 s elapsed — single launch may be too short.
    while (std::chrono::duration<double>(
               std::chrono::high_resolution_clock::now() - t0)
               .count() < 2.0) {
      warmup_compute<<<kWarmThreads / 256, 256>>>(warm_out[d], kWarmIters);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    std::printf("  d%d: warm-up done\n", d);
  }
  std::printf("\n");
  dump_gpu_state("after warm-up", num_devices);
  std::printf("\n");

  // --- (2) cudaDeviceCanAccessPeer matrix ---
  std::printf("=== cudaDeviceCanAccessPeer matrix ===\n      ");
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

  // --- allocate buffers ---
  constexpr size_t kBytes = 256u * 1024 * 1024;
  std::vector<float*> bufs(num_devices, nullptr);
  for (int d = 0; d < num_devices; d++) {
    CHECK_CUDA(cudaSetDevice(d));
    CHECK_CUDA(cudaMalloc(&bufs[d], kBytes));
  }
  void* pinned_host = nullptr;
  CHECK_CUDA(cudaMallocHost(&pinned_host, kBytes));

  // --- per-die pinned baseline ---
  std::printf("=== per-die pinned host<->device baseline (%zu MiB) ===\n",
              kBytes / 1024 / 1024);
  for (int d = 0; d < num_devices; d++) {
    CHECK_CUDA(cudaSetDevice(d));
    cudaStream_t s;
    CHECK_CUDA(cudaStreamCreate(&s));
    // warm
    CHECK_CUDA(cudaMemcpyAsync(bufs[d], pinned_host, kBytes,
                               cudaMemcpyHostToDevice, s));
    CHECK_CUDA(cudaStreamSynchronize(s));
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 4; i++) {
      CHECK_CUDA(cudaMemcpyAsync(bufs[d], pinned_host, kBytes,
                                 cudaMemcpyHostToDevice, s));
    }
    CHECK_CUDA(cudaStreamSynchronize(s));
    auto t1 = std::chrono::high_resolution_clock::now();
    double bw_h2d = (kBytes * 4.0) /
        std::chrono::duration<double>(t1 - t0).count() / 1e9;
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 4; i++) {
      CHECK_CUDA(cudaMemcpyAsync(pinned_host, bufs[d], kBytes,
                                 cudaMemcpyDeviceToHost, s));
    }
    CHECK_CUDA(cudaStreamSynchronize(s));
    auto t3 = std::chrono::high_resolution_clock::now();
    double bw_d2h = (kBytes * 4.0) /
        std::chrono::duration<double>(t3 - t2).count() / 1e9;
    std::printf("  d%d:  H2D %6.2f GB/s   D2H %6.2f GB/s\n", d, bw_h2d, bw_d2h);
    cudaStreamDestroy(s);
  }
  std::printf("\n");

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

  // --- cross-die bandwidth ---
  std::printf("=== cross-die bandwidth (%zu MiB) ===\n", kBytes / 1024 / 1024);
  constexpr int kIter = 4;
  for (int src = 0; src < num_devices; src++) {
    for (int dst = 0; dst < num_devices; dst++) {
      if (src == dst) continue;
      double bw_hw = 0.0, bw_pinned, bw_naive;
      if (peer_ok[src][dst]) {
        CHECK_CUDA(cudaSetDevice(src));
        cudaError_t e = cudaDeviceEnablePeerAccess(dst, 0);
        if (e == cudaSuccess || e == cudaErrorPeerAccessAlreadyEnabled) {
          bw_hw = bench_hw_p2p(src, dst, bufs[src], bufs[dst], kBytes, kIter);
          cudaDeviceDisablePeerAccess(dst);
        }
      }
      bw_pinned = bench_pinned_staged(src, dst, bufs[src], bufs[dst],
                                      pinned_host, kBytes, kIter);
      bw_naive = bench_peer_naive(src, dst, bufs[src], bufs[dst], kBytes, kIter);
      if (peer_ok[src][dst]) {
        std::printf("  d%d -> d%d:  hw-p2p %6.2f GB/s   pinned %6.2f GB/s   naive %6.2f GB/s\n",
                    src, dst, bw_hw, bw_pinned, bw_naive);
      } else {
        std::printf("  d%d -> d%d:  hw-p2p   N/A          pinned %6.2f GB/s   naive %6.2f GB/s\n",
                    src, dst, bw_pinned, bw_naive);
      }
    }
  }
  std::printf("\n");

  // --- final state (verifies we stayed in P0 throughout the bw test) ---
  dump_gpu_state("after measurement", num_devices);

  // cleanup
  cudaFreeHost(pinned_host);
  for (int d = 0; d < num_devices; d++) {
    cudaSetDevice(d);
    cudaFree(bufs[d]);
    cudaFree(warm_out[d]);
  }
  nvmlShutdown();

  std::printf("\nk80_p2p: probe OK\n");
  return 0;
}
