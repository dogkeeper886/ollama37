# Ollama37 Docker Build System

**Multi-stage Docker build for Ollama on CUDA 11.4 — built and tested for the Tesla K80 (compute capability 3.7); the image also compiles native CUBIN for the wider `sm_37`–`sm_86` datacenter range (Kepler→Ampere, experimental).**

## Overview

This Docker build system uses a multi-stage architecture to build and run Ollama with Tesla K80 (compute capability 3.7) support:

1. **Builder Image** (`builder/Dockerfile`) - Base environment with build tools
   - Rocky Linux 8
   - CUDA 11.4 toolkit (required for Tesla K80)
   - GCC 10 (built from source, required by CUDA 11.4)
   - CMake 4.0 (built from source)
   - Go 1.25.3

2. **Runtime Image** (`runtime/Dockerfile`) - Multi-stage build
   - **Build stage**: Compile C/C++/CUDA libraries and Go binary in builder image
   - **Runtime stage**: Copy only artifacts to slim RockyLinux 8 minimal image (~1 GB)

The runtime image contains only: ollama binary, GGML/CUDA libraries, CUDA runtime libs (cublas, cublasLt, cudart), and GCC 10 runtime libs. No compilers, build tools, or source code.

## Prerequisites

- Docker with NVIDIA Container Runtime
- Docker Compose
- NVIDIA GPU drivers (470+ for Tesla K80)
- Verify GPU access:
  ```bash
  docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
  ```

## Quick Start

### 1. Build Images

```bash
cd /home/jack/Documents/ollama37/docker
make build
```

This will:
1. Build the builder image (if not present) - **~90 minutes first time**
2. Build the runtime image - **~10 minutes**

**First-time build:** ~100 minutes total (includes building GCC 10 and CMake 4 from source)

**Subsequent builds:** ~10 minutes (builder image is cached)

### 2. Run with Docker Compose (Recommended)

```bash
docker compose up -d
```

Check logs:
```bash
docker compose logs -f
```

Stop the server:
```bash
docker compose down
```

### 3. Run Manually

```bash
docker run -d \
  --name ollama37 \
  --runtime=nvidia \
  --gpus all \
  -p 11434:11434 \
  -v ollama-data:/root/.ollama \
  ollama37:latest
```

## Usage

### Using the API

```bash
# List models
curl http://localhost:11434/api/tags

# Pull a model
curl http://localhost:11434/api/pull -d '{"name": "gemma3:4b"}'

# Run inference
curl http://localhost:11434/api/generate -d '{
  "model": "gemma3:4b",
  "prompt": "Why is the sky blue?",
  "stream": false
}'
```

### Using the CLI

```bash
# List models
docker exec ollama37 ollama list

# Pull a model
docker exec ollama37 ollama pull gemma3:4b

# Run a model
docker exec ollama37 ollama run gemma3:4b "Hello!"
```

## Architecture

### Build System Components

```
docker/
├── builder/
│   └── Dockerfile          # Base image: CUDA 11.4, GCC 10, CMake 4, Go 1.25.3
├── runtime/
│   ├── Dockerfile          # Multi-stage: build in builder, ship slim runtime (GitHub source)
│   └── Dockerfile.local    # Multi-stage: same as above but uses local source code
├── Makefile                # Build orchestration (images only)
├── docker-compose.yml      # Runtime orchestration
└── README.md               # This file
```

### Multi-Stage Build Process

#### Stage 1: Builder Image (`builder/Dockerfile`)
**Purpose**: Provide consistent build environment

**Contents:**
- Rocky Linux 8 base
- CUDA 11.4 toolkit (compilation only, no driver)
- GCC 10 from source (~60 min build time)
- CMake 4.0 from source (~8 min build time)
- Go 1.25.3 binary
- All build dependencies

**Build time:** ~90 minutes (first time), cached thereafter

**Image size:** ~15 GB

#### Stage 2: Runtime Image (`runtime/Dockerfile`)

**Build stage** (FROM ollama37-builder):
1. Clone/copy ollama37 source
2. Configure with CMake ("CUDA 11 K80" preset — native CUBIN for sm_37–sm_86, no PTX JIT)
3. Build C/C++/CUDA libraries (`cmake --build`)
4. Install artifacts to `dist/` (`cmake --install` with CPU and CUDA components)
   - Automatically bundles CUDA runtime libs (cublas, cublasLt, cudart)
   - Strips debug symbols to reduce size
5. Build Go binary

**Runtime stage** (FROM rockylinux/rockylinux:8-minimal):
1. Install ca-certificates
2. Copy ollama binary to `/usr/bin/ollama`
3. Copy GGML/CUDA libraries to `/usr/lib/ollama/`
4. Copy GCC 10 runtime libraries (libstdc++, libgcc_s) to `/usr/lib64/`
5. Configure LD_LIBRARY_PATH, NVIDIA env vars, expose ports

**Build time:** ~10 minutes

**Image size:** ~1 GB (vs ~10+ GB for single-stage)

### Library Resolution

The ollama binary at `/usr/bin/ollama` finds its libraries via `ml/path.go`:
- **Linux rule**: `{binary_dir}/../lib/ollama` = `/usr/lib/ollama/`
- **Contents**: ggml-base.so, ggml-cpu*.so, ggml-cuda.so, plus bundled CUDA runtime libs
- **GCC 10 runtime**: libstdc++.so, libgcc_s.so copied to `/usr/lib64/` (RockyLinux 8 minimal only has GCC 8's versions, which lack GLIBCXX symbols needed by GCC 10-compiled code)

## Build Commands

### Using the Makefile

```bash
# Build both builder and runtime images
make build

# Build only builder image
make build-builder

# Build only runtime image (will auto-build builder if needed)
make build-runtime

# Build runtime from local source (no GitHub clone)
make build-runtime-local

# Force rebuild without cache
make build-runtime-no-cache
make build-runtime-local-no-cache

# Remove all images
make clean

# Show help
make help
```

### Direct Docker Commands

```bash
# Build builder image
docker build -f builder/Dockerfile -t ollama37-builder:latest builder/

# Build runtime image (from GitHub)
docker build -f runtime/Dockerfile -t ollama37:latest ..

# Build runtime image (from local source)
docker build -f runtime/Dockerfile.local -t ollama37:latest ..
```

## Runtime Management

### Using Docker Compose (Recommended)

```bash
# Start server
docker compose up -d

# View logs (live tail)
docker compose logs -f

# Stop server
docker compose down

# Stop and remove volumes
docker compose down -v

# Restart server
docker compose restart
```

### Manual Docker Commands

```bash
# Start container
docker run -d \
  --name ollama37 \
  --runtime=nvidia \
  --gpus all \
  -p 11434:11434 \
  -v ollama-data:/root/.ollama \
  ollama37:latest

# View logs
docker logs -f ollama37

# Stop container
docker stop ollama37
docker rm ollama37

# Shell access (note: minimal image, limited tools available)
docker exec -it ollama37 bash
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `0.0.0.0:11434` | Server listen address |
| `LD_LIBRARY_PATH` | `/usr/lib/ollama:/usr/local/nvidia/lib:/usr/local/nvidia/lib64` | Library search path |
| `NVIDIA_VISIBLE_DEVICES` | `all` | Which GPUs to use |
| `NVIDIA_DRIVER_CAPABILITIES` | `compute,utility` | GPU capabilities |
| `OLLAMA_DEBUG` | (unset) | Enable verbose Ollama logging |
| `GGML_CUDA_DEBUG` | (unset) | Enable CUDA/CUBLAS debug logging |

### Volume Mounts

- `/root/.ollama` - Model storage (use Docker volume `ollama-data`)

### Customizing docker-compose.yml

```yaml
# Change port
ports:
  - "11435:11434"  # Host:Container

# Use specific GPU
environment:
  - NVIDIA_VISIBLE_DEVICES=0  # Use GPU 0 only

# Enable debug logging
environment:
  - OLLAMA_DEBUG=1
  - GGML_CUDA_DEBUG=1
```

## GPU Support

### Supported Compute Capabilities
- **3.7** - Tesla K80 (primary target)
- **5.0-5.2** - Maxwell (GTX 900 series)
- **6.0-6.1** - Pascal (GTX 10 series)
- **7.0-7.5** - Volta, Turing (RTX 20 series)
- **8.0-8.6** - Ampere (RTX 30 series)

### Tesla K80 Recommendations

**VRAM:** 12GB per GPU (24GB for dual-GPU K80)

**Model sizes:**
- Small (1-4B): Full precision or Q8 quantization
- Medium (7-8B): Q4_K_M quantization
- Large (13B+): Q4_0 quantization or multi-GPU

**Tested models:**
- gemma3:4b
- gpt-oss
- deepseek-r1

**Multi-GPU:**
```bash
# Use all GPUs
docker run --gpus all ...

# Use specific GPU
docker run --gpus '"device=0"' ...

# Use multiple specific GPUs
docker run --gpus '"device=0,1"' ...
```

## Troubleshooting

### GPU not detected

```bash
# Check GPU visibility in container
docker exec ollama37 nvidia-smi

# Check NVIDIA runtime
docker info | grep -i runtime
```

### NVIDIA UVM Device Files Missing

**Symptom:** `nvidia-smi` works inside the container, but Ollama reports **0 GPUs detected** (CUDA runtime cannot find GPUs).

**Root Cause:**

The nvidia-uvm device files were missing on the host system.

While the `nvidia-uvm` kernel module was loaded, the device files `/dev/nvidia-uvm` and `/dev/nvidia-uvm-tools` were not created.

These device files are critical for CUDA runtime:
- `nvidia-smi` only needs the basic driver (works without UVM)
- **CUDA applications require UVM** for GPU memory allocation and kernel execution
- Without UVM devices: CUDA reports 0 GPUs even though they exist

**The Fix:**

Run this single command on the **host system** (not inside the container):

```bash
nvidia-modprobe -u -c=0
```

This creates the required device files:
- `/dev/nvidia-uvm` (major 239, minor 0)
- `/dev/nvidia-uvm-tools` (major 239, minor 1)

Then restart the container:

```bash
docker compose restart
```

**Result:** GPUs now properly detected by CUDA runtime.

**Verify the fix:**

```bash
# Check UVM device files exist on host
ls -l /dev/nvidia-uvm*

# Check Ollama logs for GPU detection
docker compose logs | grep -i gpu

# You should see output like:
# ollama37  | time=... level=INFO msg="Nvidia GPU detected" name="Tesla K80" vram=11441 MiB
# ollama37  | time=... level=INFO msg="Nvidia GPU detected" name="Tesla K80" vram=11441 MiB
```

### Model fails to load

```bash
# Check logs with CUDA debug
docker run --rm --runtime=nvidia --gpus all \
  -e OLLAMA_DEBUG=1 \
  -e GGML_CUDA_DEBUG=1 \
  -p 11434:11434 \
  ollama37:latest

# Check library paths
docker exec ollama37 bash -c 'echo $LD_LIBRARY_PATH'

# Verify all library dependencies resolve (install ldd if needed)
docker exec ollama37 bash -c 'ls /usr/lib/ollama/'
```

### Verifying runtime image dependencies

After building, verify the runtime image has all required libraries:

```bash
# Check that all .so files can resolve their dependencies
docker run --rm ollama37:latest bash -c \
  'for f in /usr/lib/ollama/*.so; do echo "=== $f ==="; ldd "$f" 2>&1 | grep "not found" || echo "OK"; done'

# Check ollama binary dependencies
docker run --rm ollama37:latest ldd /usr/bin/ollama

# Check image size
docker images ollama37:latest
```

### Build fails with "out of memory"

```bash
# Edit runtime/Dockerfile line for cmake build
# Change: cmake --build build -j$(nproc)
# To: cmake --build build -j2

# Or set Docker memory limit
docker build --memory=8g ...
```

### Port already in use

```bash
# Find process using port 11434
sudo lsof -i :11434

# Kill the process or change port in docker-compose.yml
ports:
  - "11435:11434"
```

### Build cache issues

```bash
# Rebuild runtime image without cache
make build-runtime-no-cache

# Rebuild builder image without cache
docker build --no-cache -f builder/Dockerfile -t ollama37-builder:latest builder/

# Remove all images and rebuild
make clean
make build
```

## Rebuilding

### Rebuild with latest code

```bash
# Runtime Dockerfile clones from GitHub, so rebuild to get latest
make build-runtime

# Restart container
docker compose restart
```

### Rebuild everything from scratch

```bash
# Stop and remove containers
docker compose down -v

# Remove images
make clean

# Rebuild all
make build

# Start fresh
docker compose up -d
```

### Rebuild only builder (rare)

```bash
# Only needed if you change CUDA/GCC/CMake/Go versions
make clean
make build-builder
make build-runtime
```

## Development

### Modifying the build

1. **Change build tools** - Edit `builder/Dockerfile`
2. **Change Ollama build process** - Edit `runtime/Dockerfile` or `runtime/Dockerfile.local`
3. **Change build orchestration** - Edit `Makefile`
4. **Change runtime config** - Edit `docker-compose.yml`

### Testing changes locally

```bash
# Build with local source (no need to push to GitHub)
make build-runtime-local

# Run and test
docker compose up -d
docker compose logs -f
```

### Debugging inside the container

The runtime image is minimal (no compilers or debug tools). For debugging:

```bash
# Check what's in the runtime image
docker exec ollama37 ls /usr/lib/ollama/
docker exec ollama37 /usr/bin/ollama --version

# For full debugging, run the builder image directly
docker run --rm -it --runtime=nvidia --gpus all ollama37-builder bash
```

## Image Sizes

| Image | Size | Contents |
|-------|------|----------|
| `ollama37-builder:latest` | ~15 GB | CUDA, GCC, CMake, Go, build deps |
| `ollama37:latest` | ~1 GB | Ollama binary + GGML/CUDA libraries only |

## Build Times

| Task | First Build | Cached Build |
|------|-------------|--------------|
| Builder image | ~90 min | <1 min |
| Runtime image | ~10 min | ~10 min |
| **Total** | **~100 min** | **~10 min** |

**Breakdown (first build):**
- GCC 10: ~60 min
- CMake 4: ~8 min
- CUDA toolkit: ~10 min
- Go install: ~1 min
- Ollama build: ~10 min

## Documentation

- **[../CLAUDE.md](../CLAUDE.md)** - Project goals, implementation details, and technical notes
- **[Upstream Ollama](https://github.com/ollama/ollama)** - Original Ollama project
- **[dogkeeper886/ollama37](https://github.com/dogkeeper886/ollama37)** - This fork with K80 support

## License

MIT (same as upstream Ollama)
