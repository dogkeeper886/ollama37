# Ollama37 Runtime Image
# Multi-stage build: compiles in builder image, copies only runtime artifacts to slim image
#
# Image size: ~800 MB - 1.5 GB (vs ~8-12 GB for single-stage builder image)
# The runtime image contains only: ollama binary, GGML/CUDA libraries, CUDA runtime
# libs, and GCC 10 runtime libs. No compilers, build tools, or source code.

# === BUILD STAGE ===
# Builder image: ollama37-builder contains GCC 10, CUDA 11.4, and build tools
FROM ollama37-builder AS builder

# GIT_COMMIT busts Docker cache when new commits are pushed to GitHub.
# Without this, Docker caches the git clone layer and never picks up new code.
ARG GIT_COMMIT=HEAD
# Clone ollama37 source code from GitHub
RUN cd /usr/local/src \
    && git clone https://github.com/dogkeeper886/ollama37.git \
    && cd ollama37 \
    && git checkout ${GIT_COMMIT}

# Set working directory for build
WORKDIR /usr/local/src/ollama37

# Configure build with CMake
# Use "CUDA 11 K80" preset: compiles native CUBIN for the 470-era datacenter sweep
# (sm_37/50/52/60/61/70/75/80/86 — Kepler through Ampere), all -real so there's no
# PTX JIT on container restart. Each GPU loads only its own cubin, so the K80 path is
# byte-identical; the extra cubins are dead weight other cards never touch.
# The "CUDA 11" preset embeds PTX instead, requiring ~3-4 min of JIT compilation on
# every container restart. Native CUBIN eliminates that.
# Set LD_LIBRARY_PATH during build so CMake can locate GCC 10 runtime libraries
# and properly link against them (required for C++ standard library and atomics)
RUN bash -c 'LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64:/usr/lib64:$LD_LIBRARY_PATH \
    CC=/usr/local/bin/gcc CXX=/usr/local/bin/g++ \
    cmake --preset "CUDA 11 K80"'

# Build C/C++/CUDA libraries with CMake
# Compile all GGML CUDA kernels and Ollama native libraries
# Use all available CPU cores (-j) for parallel compilation to speed up build
RUN bash -c 'LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64:/usr/lib64:$LD_LIBRARY_PATH \
    CC=/usr/local/bin/gcc CXX=/usr/local/bin/g++ \
    cmake --build build -j$(nproc)'

# Install compiled libraries to dist/ directory
# cmake --install collects runtime artifacts and their dependencies:
#   CPU component: ggml-base.so, ggml-cpu*.so variants
#   CUDA component: ggml-cuda.so + bundled CUDA runtime libs (cublas, cublasLt, cudart)
# --strip removes debug symbols to reduce library sizes
# Install destination is controlled by CMakePresets.json installDir: ${sourceDir}/dist
RUN cmake --install build --component CPU --strip \
    && cmake --install build --component CUDA --strip

# Build Go binary
# VCS info is embedded automatically since we cloned from git
# Set version via -ldflags - pass OLLAMA_VERSION as build arg
ARG OLLAMA_VERSION
RUN mkdir -p dist/bin \
    && go build -ldflags "-X github.com/ollama/ollama/version.Version=${OLLAMA_VERSION}" -o dist/bin/ollama .

# === RUNTIME STAGE ===
# Slim base image with only runtime dependencies
# RockyLinux 8 minimal: same OS family as builder, glibc-compatible
FROM rockylinux/rockylinux:8-minimal

# Install minimal runtime dependencies
RUN microdnf install -y ca-certificates && microdnf clean all

# Copy the ollama binary
# Placed at /usr/bin/ollama so ml/path.go resolves libs at ../lib/ollama = /usr/lib/ollama
COPY --from=builder /usr/local/src/ollama37/dist/bin/ollama /usr/bin/ollama

# Copy compiled GGML/CUDA libraries (includes bundled CUDA runtime libs)
# cmake --install already collected: ggml-base.so, ggml-cpu*.so, ggml-cuda.so,
# libcublas.so, libcublasLt.so, libcudart.so
COPY --from=builder /usr/local/src/ollama37/dist/lib/ollama /usr/lib/ollama

# Copy GCC 10 runtime libraries from the builder
# These are custom-built in the builder (not system packages) and required by all compiled .so files
# RockyLinux 8 minimal only has GCC 8's libstdc++ which lacks GLIBCXX symbols from GCC 10
COPY --from=builder /usr/local/lib64/libstdc++.so* /usr/lib64/
COPY --from=builder /usr/local/lib64/libgcc_s.so* /usr/lib64/

# Library search paths:
#   /usr/lib/ollama - GGML/CUDA .so files and bundled CUDA runtime libs
#   /usr/local/nvidia/lib{64} - NVIDIA container runtime driver mount
ENV LD_LIBRARY_PATH=/usr/lib/ollama:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# NVIDIA container runtime configuration
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NVIDIA_VISIBLE_DEVICES=all

# Configure Ollama server to listen on all interfaces
ENV OLLAMA_HOST=0.0.0.0:11434

# Expose Ollama API port
EXPOSE 11434

# Create persistent volume for model storage
# Models downloaded by Ollama will be stored here
RUN mkdir -p /root/.ollama
VOLUME ["/root/.ollama"]

# Configure health check to verify Ollama is running
# Uses 'ollama list' command to check if the service is responsive
# This validates both API availability and model registry access
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD /usr/bin/ollama list || exit 1

# Container runs 'ollama serve' by default to start the API server
ENTRYPOINT ["/usr/bin/ollama"]
CMD ["serve"]
