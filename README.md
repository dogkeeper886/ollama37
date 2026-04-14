# Ollama37

**Ollama fork with CUDA compute capability 3.7 support for Tesla K80 GPUs**

## Demo Video

[![Why I Run AI on 10-Year-Old GPUs](https://img.youtube.com/vi/iYxgGsPu5rM/maxresdefault.jpg)](https://www.youtube.com/watch?v=iYxgGsPu5rM)

**[Why I Run AI on 10-Year-Old GPUs: Ollama K80 Docker Build System & CI/CD Pipeline](https://www.youtube.com/watch?v=iYxgGsPu5rM)**

In this video, I walk through our modern AI CI/CD pipeline built entirely on legacy Nvidia K80 GPUs. Why use old hardware in 2025? Because the hands-on challenges teach you skills you'd never learn otherwise—compiling kernels, managing GCC versions, debugging driver issues, and navigating end-of-life software dependencies.

**Highlights:**
- **Docker Build System**: Multi-stage build with Rocky Linux 8, CUDA 11.4, GCC 10, CMake 4.0, Go 1.25.3
- **CI/CD & Test Framework**: Self-hosted GitHub Actions runner with Simple Judge (exit code/pattern matching) and LLM Judge (Gemma3 12B for log analysis)
- **Full Pipeline**: Build → Container check → Runtime verification → Inference testing → Model compatibility testing

## Overview

This fork adds CUDA compute capability 3.7 to Ollama, enabling it to run on Tesla K80 GPUs. It uses a multi-stage Docker build system that compiles everything from source and produces a slim ~1 GB runtime image.

### Environment
- Rocky Linux 8
- CUDA 11.4 toolkit
- GCC 10 (built from source)
- CMake 4.0 (built from source)
- Go 1.25.3
- NVIDIA driver 470+

### Supported GPUs
- **3.7** - Tesla K80 (primary target)
- **5.0-8.6** - Maxwell through Ampere (GTX 900 series to RTX 30 series)

### Tesla K80 Model Recommendations

**VRAM:** 12GB per GPU (24GB for dual-GPU K80)

| Model Size | Quantization |
|------------|-------------|
| Small (1-4B) | Full precision or Q8 |
| Medium (7-8B) | Q4_K_M |
| Large (13B+) | Q4_0 or multi-GPU |

## Quick Start

```bash
# Build
cd docker
make build

# Run
docker compose up -d

# Test
curl http://localhost:11434/api/tags
docker exec ollama37 ollama pull gemma3:4b
docker exec ollama37 ollama run gemma3:4b "Hello!"
```

## Documentation

- **[docker/README.md](docker/README.md)** — Full Docker build system docs (architecture, build commands, configuration, troubleshooting)
- **[CLAUDE.md](CLAUDE.md)** — Development process, branch workflow, and Claude Code instructions
- **[Upstream Ollama](https://github.com/ollama/ollama)** — Original Ollama project

## License

MIT (same as upstream Ollama)
