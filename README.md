# Ollama37

**Run modern LLMs on the NVIDIA Tesla K80.** This repository is a fork of
[ollama37](https://github.com/dogkeeper886/ollama37) — itself a fork of
[Ollama](https://github.com/ollama/ollama) that restores CUDA compute capability
3.7 — so a 2014-era K80 can serve today's models (Gemma, Qwen, DeepSeek, and more)
long after upstream dropped Kepler support.

**Upstream sync v0.40.16.** The `sync/upstream-v0.40.16` branch merges upstream
tag `v0.40.16` (llama.cpp `b12177`) with every K80 semantic re-applied on top, and
it becomes the new `main` of this fork once the pull request is merged. See
[Upstream sync v0.40.16](#upstream-sync-v04016).

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Docker image](https://img.shields.io/docker/v/dogkeeper886/ollama37?label=docker%20hub&sort=semver)](https://hub.docker.com/r/dogkeeper886/ollama37)
[![Docker pulls](https://img.shields.io/docker/pulls/dogkeeper886/ollama37.svg)](https://hub.docker.com/r/dogkeeper886/ollama37)

![Ollama37 brings modern LLMs to the Tesla K80](docs/images/readme/concept.png)

> [!IMPORTANT]
> **The Tesla K80 (compute capability 3.7) is the only tested target** — the hardware this
> project is built and tuned for. The published image now also compiles native CUBIN
> for the whole 470-era datacenter range (`sm_37`–`sm_86`, Kepler through Ampere), so a
> P100, V100, T4, A100 and similar cards *should* load it — but those are **not yet
> hardware-validated, so treat them as experimental**. Cards outside that range (Ada,
> Hopper) need a self-build — see [Building for other GPUs](#building-for-other-gpus).

## Contents

- [Why Ollama37](#why-ollama37)
- [Features](#features)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Tested Models](#tested-models)
- [How It's Built](#how-its-built)
- [Upstream sync v0.40.16](#upstream-sync-v04016)
- [Building for other GPUs](#building-for-other-gpus)
- [Troubleshooting](#troubleshooting)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Why Ollama37

The Tesla K80 is a 24 GB (2×12 GB) datacenter GPU you can buy used for very little, but
modern Ollama won't run on it: CUDA 12, recent drivers, and upstream's prebuilt binaries
all dropped Kepler (`sm_37`). Ollama37 keeps that hardware useful by pinning the exact
toolchain the K80 still needs and shipping a ready-to-run image. Doing this by hand also
teaches the parts that "just works" hides — compiling CUDA kernels, matching GCC to CUDA,
and wrangling end-of-life drivers.

## Features

- **Tesla K80 support** — full CUDA compute capability 3.7 (`sm_37`), the tested target.
- **Broader datacenter range (experimental)** — the image also compiles native CUBIN for
  `sm_50`–`sm_86` (Maxwell through Ampere: P100, V100, T4, A100, …), so they can load it,
  though only the K80 is hardware-validated.
- **Fast cold start** — compiled to native CUBIN, so there's no multi-minute PTX JIT on
  container start (a PTX build re-JITs for ~3–4 min on every restart).
- **Qwen3.5 DeltaNet** — first Ollama fork to support the DeltaNet recurrent architecture.
- **Tool calling** — verified via the API and LangChain.
- **Synced to upstream v0.40.16** — llama.cpp `b12177` engine with all K80
  semantics re-applied (see [Upstream sync v0.40.16](#upstream-sync-v04016)).

## Requirements

- A **Tesla K80** (the tested target), or another NVIDIA datacenter card in the
  `sm_37`–`sm_86` range (Maxwell→Ampere) — experimental, see the note at the top.
- **NVIDIA driver 470+** on the host. 470 is the last branch that supports Kepler —
  newer drivers dropped the K80.
- **Docker** with the **NVIDIA Container Runtime**.

The container ships no driver of its own. At run time the NVIDIA Container Runtime mounts
the host's driver in via `--gpus`, which is how the image reaches the K80:

![Host driver → NVIDIA Container Runtime → ollama37 image → Tesla K80](docs/images/readme/runtime-stack.png)

Why those exact versions? The K80's age pins the entire chain — newer drivers, CUDA, and
GCC have all removed Kepler support:

![K80 needs driver 470, which pairs with CUDA 11.4, which caps GCC at 10](docs/images/readme/version-chain.png)

## Quick Start

![Pull the image, start the server, pull a model, then chat](docs/images/readme/quickstart-flow.png)

Pull the prebuilt image from Docker Hub and run it (no build required):

```bash
docker run -d \
  --name ollama37 \
  --runtime=nvidia \
  --gpus all \
  -p 11434:11434 \
  -v ollama-data:/root/.ollama \
  dogkeeper886/ollama37:latest
```

Then pull a model and chat:

```bash
docker exec ollama37 ollama pull gemma3:4b
docker exec ollama37 ollama run gemma3:4b "Why is the sky blue?"
```

Or call the API:

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "gemma3:4b",
  "prompt": "Why is the sky blue?",
  "stream": false
}'
```

For Docker Compose and the full configuration, see [docker/README.md](docker/README.md).

## Tested Models

![Small models run at full precision or Q8, medium at Q4_K_M, large at Q4_0 or multi-GPU](docs/images/readme/model-sizing.png)

A K80 has **12 GB VRAM per GPU** (24 GB for a dual-GPU board). Size quantization to fit:

| Model size    | Quantization         |
|---------------|----------------------|
| Small (1–4B)  | Full precision or Q8 |
| Medium (7–8B) | Q4_K_M               |
| Large (13B+)  | Q4_0 or multi-GPU    |

Verified on K80: `gemma3:4b`, `gemma3:27b`, `gemma4:e4b`, `gemma4:26b`, `qwen3.5:9b`,
`qwen3.5:27b`, `qwen3.8:27b`, `gpt-oss:20b`, `deepseek-r1:7b`, `ministral-3`, `functiongemma`.
The v0.40.16 sync matrix additionally ran on K80: `bonsai-2:4b`, `bonsai-2:9b`,
`qwen3.5:4b`, `granite4.2:8b`, Mellum2.1-12B-A2.5B (MoE), `ifm/k2-horizon:7b`,
`ornith-1.5:9b`, `nemotron-3.5-bolt:e4b` (see the
[benchmark matrix](#k80-benchmark-matrix)).

## How It's Built

Ollama37 is split into two Docker images so the thing you run stays small:

![Source compiles in a ~15 GB builder image into a ~1 GB runtime image, published to Docker Hub](docs/images/readme/build-pipeline.png)

- **`ollama37-builder` (~15 GB)** — the build environment only: CUDA 11.4 toolkit, GCC 10
  and CMake 4 (compiled from source, ~90 min the first time), and Go. It is never shipped.
- **`ollama37` (~1 GB)** — a multi-stage build. Two CMake builds run *inside* the builder
  (upstream's split: the `cpu` preset installs `llama-server` and the core libraries, the
  `llama_cuda_v11_linux` preset installs the GPU backend with native CUBIN for `sm_37`–`sm_86`,
  the 470 datacenter sweep), then only the
  artifacts — the `ollama` binary, the GGML/CUDA libraries, the bundled CUDA runtime libs
  (cuBLAS, cuBLASLt, cudart), and the GCC 10 runtime libs — are copied onto a slim
  `rockylinux:8-minimal` base. This is the image published to Docker Hub.

To build it yourself:

```bash
cd docker
make build          # builds the builder image (first time only), then the runtime image
docker compose up -d
```

## Upstream sync v0.40.16

The fork tracks one immutable upstream tag per release line. The
`sync/upstream-v0.40.16` branch merges upstream tag **`v0.40.16`** (`d63b70260`)
on top of the previous sync point (**`v0.40.13`**, llama.cpp `b11963`) and bumps
the pinned engine to **llama.cpp `b12177`** — and it becomes the new `main` of
this fork once the pull request is merged.

- **Structural merge** resolved per the documented path policy
  ([`docs/upstream-sync.md`](docs/upstream-sync.md) §3): the fork's workflows and
  two-build payload are kept, and every auto-merged Go/CMake file was verified
  hunk by hunk to preserve the fork's semantics.
- **Go semantics re-applied** — flash-attention gate, graph safety multiplier,
  `/api/metrics` memory components, K80 model templates.
- **`llama/compat-k80/` re-audited for `b12177`** — the full chain (`001` +
  `1000`–`1006`) applies cleanly on a fresh llama.cpp clone; only `1001` needed
  re-basing (a new `ggml-cuda` include shifted its hunk), and `ggml-cuda` compiles
  green (nvcc 11.4.3, `sm_37`, `GGML_CUDA_NCCL=OFF`).
- **New in `b12177`: the `GGML_CUDA_NCCL` allreduce path** — ON by default, needs
  NCCL ≥ 2.17. The builder image ships no NCCL, so the option degrades to a
  warning and is inert for the K80 payload; future builds must stay NCCL-free or
  pass `-DGGML_CUDA_NCCL=OFF`.

**Gates.** `go build` / `go vet` / `go test` green (53 packages; the MLX stack is
excluded — a pre-existing builder link issue, not a regression). Full local
payload: CPU preset + `llama_cuda_v11_linux` (`sm_37`–`sm_86` CUBIN) + Go binary,
reporting `ollama version is 2.4.0-sync`.

### K80 benchmark matrix

Measured on a dedicated K80 die (11.4 GiB, driver 470.256.02), away from the
production instance: throwaway container per run, `OLLAMA_KEEP_ALIVE=0` (fresh
load per request), `OLLAMA_KV_CACHE_TYPE=f16`, `OLLAMA_NUM_PARALLEL=1`,
`num_ctx=4096`, temperature 0, thinking off, fixed ~40-token prompt,
128-token generations, timings from the API payload. The K80's 149 W cap
throttles clocks under sustained load, so the matrix was thermal-controlled —
back-to-back runs on the same die, order alternated across models, 20 s cooldown
between pairs.

| Model | Quant | Load (s) | Decode (t/s) | Prefill (t/s) |
|---|---|---|---|---|
| bonsai-2:4b | Q2_0 | 4.1 | 17.66 | 28.4 |
| qwen3.5:4b | Q4_K_M | 5.1 | 11.65 | 27.3 |
| granite4.2:8b | Q4_K_M | 4.8 | 6.93 | 19.9 |
| bonsai-2:9b | Q2_0 | 4.8 | 8.79 | 16.6 |
| hf.co/JetBrains/Mellum2.1-12B-A2.5B (MoE) | Q4_K_M | 6.0 | 15.64 | 27.9 |
| ifm/k2-horizon:7b | Q4_K_M | 4.9 | 6.47 | 16.2 |
| ornith-1.5:9b | Q4_K_M | 6.9 | 6.38 | 15.9 |
| nemotron-3.5-bolt:e4b | Q4_K_M | 7.8 | 2.20 | 14.1 |

**Notes.** `k2-horizon:7b` and `ornith-1.5:9b` decode about 1.9× faster than on
the previous line (`v0.40.13`) — a genuine llama.cpp `b11963`→`b12177` engine
gain. `nemotron-3.5-bolt:e4b` regressed on the new line (MoE decode path) and is
flagged for a follow-up issue before release.

**Before merge to `main`.** sm37 CI canary + full pipeline (build → deploy →
canary → models → throughput → mcp) on the sm37 runner, and the
`nemotron-3.5-bolt:e4b` follow-up. Rollback assets are verified pullable
(`ollama37:sync-v0.40.13-c293a5a6`, `ollama37:pre-sync-rollback`); the agent
judge pin is unchanged.

## Building for other GPUs

The published image now targets the whole 470-era datacenter range (`sm_37` through
`sm_86`, Kepler through Ampere), but **only the K80 is tested** — the other architectures
are compiled in and should load, yet are unvalidated, so you are on your own there.

For cards **outside** that range — Ada (RTX 40-series) and Hopper (H100), which need
CUDA 12+ and a newer driver — you have to build it yourself: `CMakePresets.json` ships
`CUDA 12` / `CUDA 13` presets for them. Note that moving to a newer CUDA/driver **drops
K80 support** — the very thing this project pins CUDA 11.4 / driver 470 to keep. See
[docker/README.md](docker/README.md).

## Troubleshooting

The most common K80 snag: `nvidia-smi` works inside the container but Ollama reports **0
GPUs**, because the host is missing the UVM device files. Fix it on the host with:

```bash
nvidia-modprobe -u -c=0
```

This and other issues (GPU not detected, model load failures, build problems) are covered
in [docker/README.md](docker/README.md#troubleshooting).

## Documentation

- **[docker/README.md](docker/README.md)** — full build system: architecture, build
  commands, configuration, and troubleshooting.
- **[docs/upstream-sync.md](docs/upstream-sync.md)** — how the fork is kept in sync
  with upstream Ollama tags, and the `llama/compat-k80/` patch workflow.
- **[CLAUDE.md](CLAUDE.md)** — development process and contribution workflow.
- **[Upstream Ollama](https://github.com/ollama/ollama)** — the original project.

## Contributing

Issues and pull requests are welcome — Tesla K80 compatibility reports and fixes
especially. See the [issue tracker](https://github.com/dogkeeper886/ollama37/issues).

## License

MIT, the same as upstream Ollama. See [LICENSE](LICENSE).
