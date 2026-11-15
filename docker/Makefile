# Makefile for Ollama37 Docker Build System
#
# This Makefile manages the Docker build process:
#   1. Builder image: Base environment with CUDA 11.4, GCC 10, CMake 4, Go 1.26.10
#   2. Runtime image: Multi-stage build that compiles in builder, ships slim runtime
#
# The runtime Dockerfile handles:
#   - Build stage: clone/copy source, compile C/C++/CUDA/Go in builder image
#   - Runtime stage: copy only binary + libraries to slim RockyLinux 8 minimal image
#   - Result: ~1 GB image instead of ~10 GB (no compilers, build tools, or source)
#
# Usage:
#   make build          - Build builder and runtime images (default)
#   make build-builder  - Build only the builder image
#   make build-runtime  - Build only the runtime image
#   make clean          - Remove all Docker images
#   make help           - Show help message
#
# To run the container, use docker-compose:
#   docker-compose up -d
#   docker-compose logs -f
#   docker-compose down

# Load .env file if it exists (for local development)
-include .env

# Configuration
BUILDER_IMAGE := ollama37-builder
BUILDER_TAG := latest
RUNTIME_IMAGE := ollama37
RUNTIME_TAG := latest
SOURCE_DIR := $(shell cd .. && pwd)
# Version: environment variable > .env file > fallback 0.0.0
OLLAMA_VERSION ?= 0.0.0
BUILDER_DOCKERFILE := $(SOURCE_DIR)/docker/builder/Dockerfile
RUNTIME_DOCKERFILE := $(SOURCE_DIR)/docker/runtime/Dockerfile

# Docker build context directories
BUILDER_CONTEXT := $(SOURCE_DIR)/docker/builder
RUNTIME_CONTEXT := $(SOURCE_DIR)

# Dockerfiles
RUNTIME_DOCKERFILE_LOCAL := $(SOURCE_DIR)/docker/runtime/Dockerfile.local

.PHONY: all build build-builder build-runtime build-runtime-no-cache build-runtime-local build-runtime-local-no-cache ensure-builder clean help

# Default target
all: build

# Build both builder and runtime images
build: build-builder build-runtime
	@echo ""
	@echo "✓ All images built successfully!"
	@echo "  Builder: $(BUILDER_IMAGE):$(BUILDER_TAG)"
	@echo "  Runtime: $(RUNTIME_IMAGE):$(RUNTIME_TAG)"
	@echo ""
	@echo "To start the Ollama server:"
	@echo "  docker-compose up -d"
	@echo ""
	@echo "View logs:"
	@echo "  docker-compose logs -f"
	@echo ""
	@echo "Stop the server:"
	@echo "  docker-compose down"

# Build the builder base image
build-builder:
	@echo "→ Building builder image..."
	@echo "  Image: $(BUILDER_IMAGE):$(BUILDER_TAG)"
	@echo "  Dockerfile: $(BUILDER_DOCKERFILE)"
	@echo ""
	@docker build \
		-f $(BUILDER_DOCKERFILE) \
		-t $(BUILDER_IMAGE):$(BUILDER_TAG) \
		$(BUILDER_CONTEXT)
	@echo ""
	@echo "✓ Builder image built successfully!"

# Build the runtime image (requires builder image)
build-runtime: ensure-builder
	@echo "→ Building runtime image..."
	@echo "  Image: $(RUNTIME_IMAGE):$(RUNTIME_TAG)"
	@echo "  Dockerfile: $(RUNTIME_DOCKERFILE)"
	@echo "  Source: GitHub (cache-busted by commit hash)"
	@echo "  Commit: $(shell git rev-parse HEAD)"
	@echo ""
	@echo "  This will:"
	@echo "    - Clone ollama37 source from GitHub"
	@echo "    - Configure with CMake (llama_cuda_v11_linux preset)"
	@echo "    - Compile C/C++/CUDA libraries"
	@echo "    - Build Go binary"
	@echo "    - Package slim runtime image (~1 GB)"
	@echo ""
	@docker build \
		--build-arg OLLAMA_VERSION=$(OLLAMA_VERSION) \
		--build-arg GIT_COMMIT=$(shell git rev-parse HEAD) \
		-f $(RUNTIME_DOCKERFILE) \
		-t $(RUNTIME_IMAGE):$(RUNTIME_TAG) \
		$(RUNTIME_CONTEXT)
	@echo ""
	@echo "✓ Runtime image built successfully!"
	@echo ""
	@echo "To start the Ollama server:"
	@echo "  docker-compose up -d"

# Build runtime with --no-cache (force fresh GitHub clone)
build-runtime-no-cache: ensure-builder
	@echo "→ Building runtime image (no cache)..."
	@echo "  Image: $(RUNTIME_IMAGE):$(RUNTIME_TAG)"
	@echo "  Dockerfile: $(RUNTIME_DOCKERFILE)"
	@echo "  Source: GitHub (fresh clone, no cache)"
	@echo "  Version: $(OLLAMA_VERSION)"
	@echo ""
	@echo "  This will:"
	@echo "    - Force fresh clone from GitHub"
	@echo "    - Rebuild all layers without cache"
	@echo "    - Configure with CMake (llama_cuda_v11_linux preset)"
	@echo "    - Compile C/C++/CUDA libraries"
	@echo "    - Build Go binary with version $(OLLAMA_VERSION)"
	@echo "    - Package slim runtime image (~1 GB)"
	@echo ""
	@docker build --no-cache \
		--build-arg OLLAMA_VERSION=$(OLLAMA_VERSION) \
		--build-arg GIT_COMMIT=$(shell git rev-parse HEAD) \
		-f $(RUNTIME_DOCKERFILE) \
		-t $(RUNTIME_IMAGE):$(RUNTIME_TAG) \
		$(RUNTIME_CONTEXT)
	@echo ""
	@echo "✓ Runtime image built successfully!"
	@echo ""
	@echo "To start the Ollama server:"
	@echo "  docker-compose up -d"

# Build runtime using local source code
build-runtime-local: ensure-builder
	@echo "→ Building runtime image (local source)..."
	@echo "  Image: $(RUNTIME_IMAGE):$(RUNTIME_TAG)"
	@echo "  Dockerfile: $(RUNTIME_DOCKERFILE_LOCAL)"
	@echo "  Source: Local directory (uses cache)"
	@echo "  Version: $(OLLAMA_VERSION)"
	@echo ""
	@echo "  This will:"
	@echo "    - Copy local source code to container"
	@echo "    - Configure with CMake (llama_cuda_v11_linux preset)"
	@echo "    - Compile C/C++/CUDA libraries"
	@echo "    - Build Go binary with version $(OLLAMA_VERSION)"
	@echo "    - Package slim runtime image (~1 GB)"
	@echo ""
	@docker build \
		--build-arg OLLAMA_VERSION=$(OLLAMA_VERSION) \
		--build-arg GIT_COMMIT=$(shell git rev-parse HEAD) \
		-f $(RUNTIME_DOCKERFILE_LOCAL) \
		-t $(RUNTIME_IMAGE):$(RUNTIME_TAG) \
		$(RUNTIME_CONTEXT)
	@echo ""
	@echo "✓ Runtime image built successfully!"
	@echo ""
	@echo "To start the Ollama server:"
	@echo "  docker-compose up -d"

# Build runtime using local source with --no-cache
build-runtime-local-no-cache: ensure-builder
	@echo "→ Building runtime image (local source, no cache)..."
	@echo "  Image: $(RUNTIME_IMAGE):$(RUNTIME_TAG)"
	@echo "  Dockerfile: $(RUNTIME_DOCKERFILE_LOCAL)"
	@echo "  Source: Local directory (no cache)"
	@echo "  Version: $(OLLAMA_VERSION)"
	@echo ""
	@echo "  This will:"
	@echo "    - Copy local source code to container"
	@echo "    - Rebuild all layers without cache"
	@echo "    - Configure with CMake (llama_cuda_v11_linux preset)"
	@echo "    - Compile C/C++/CUDA libraries"
	@echo "    - Build Go binary with version $(OLLAMA_VERSION)"
	@echo "    - Package slim runtime image (~1 GB)"
	@echo ""
	@docker build --no-cache \
		--build-arg OLLAMA_VERSION=$(OLLAMA_VERSION) \
		--build-arg GIT_COMMIT=$(shell git rev-parse HEAD) \
		-f $(RUNTIME_DOCKERFILE_LOCAL) \
		-t $(RUNTIME_IMAGE):$(RUNTIME_TAG) \
		$(RUNTIME_CONTEXT)
	@echo ""
	@echo "✓ Runtime image built successfully!"
	@echo ""
	@echo "To start the Ollama server:"
	@echo "  docker-compose up -d"

# Ensure builder image exists (build if not present)
ensure-builder:
	@if ! docker images --format '{{.Repository}}:{{.Tag}}' | grep -q "^$(BUILDER_IMAGE):$(BUILDER_TAG)$$"; then \
		echo "→ Builder image not found. Building $(BUILDER_IMAGE):$(BUILDER_TAG)..."; \
		echo ""; \
		$(MAKE) build-builder; \
		echo ""; \
	fi

# Remove all Docker images
clean:
	@echo "→ Removing Docker images..."
	@docker rmi $(RUNTIME_IMAGE):$(RUNTIME_TAG) 2>/dev/null || echo "  No runtime image to remove"
	@docker rmi $(BUILDER_IMAGE):$(BUILDER_TAG) 2>/dev/null || echo "  No builder image to remove"
	@echo ""
	@echo "✓ Images removed"
	@echo ""
	@echo "Note: To remove containers and volumes, use:"
	@echo "  docker-compose down -v"

# Show help message
help:
	@echo "Ollama37 Docker Build System"
	@echo ""
	@echo "Build Targets:"
	@echo "  make build                        - Build builder and runtime images (default)"
	@echo "  make build-builder                - Build only the builder base image"
	@echo "  make build-runtime                - Build runtime from GitHub (uses cache)"
	@echo "  make build-runtime-no-cache       - Build runtime from GitHub (fresh clone, no cache)"
	@echo "  make build-runtime-local          - Build runtime from local source (uses cache)"
	@echo "  make build-runtime-local-no-cache - Build runtime from local source (no cache)"
	@echo "  make clean                        - Remove all Docker images"
	@echo "  make help                         - Show this help message"
	@echo ""
	@echo "Which Build Target to Use?"
	@echo "  • build-runtime               - Normal builds after pushing to GitHub"
	@echo "  • build-runtime-no-cache      - After GitHub push when Docker cache is stale"
	@echo "  • build-runtime-local         - Quick testing of local changes without push"
	@echo "  • build-runtime-local-no-cache- Full rebuild with local changes"
	@echo ""
	@echo "Configuration:"
	@echo "  BUILDER_IMAGE:   $(BUILDER_IMAGE):$(BUILDER_TAG)"
	@echo "  RUNTIME_IMAGE:   $(RUNTIME_IMAGE):$(RUNTIME_TAG)"
	@echo ""
	@echo "Dockerfiles:"
	@echo "  Builder:         $(BUILDER_DOCKERFILE)"
	@echo "  Runtime (GitHub):$(RUNTIME_DOCKERFILE)"
	@echo "  Runtime (Local): $(RUNTIME_DOCKERFILE_LOCAL)"
	@echo ""
	@echo "Build Architecture:"
	@echo "  1. Builder image: Base environment (CUDA 11.4, GCC 10, CMake 4, Go 1.26.10)"
	@echo "  2. Runtime image: Multi-stage build (compile in builder, ship slim runtime)"
	@echo "     - Build stage: compile C/C++/CUDA/Go in builder image"
	@echo "     - Runtime stage: copy artifacts to RockyLinux 8 minimal (~1 GB vs ~10 GB)"
	@echo ""
	@echo "Container Management (use docker-compose):"
	@echo "  docker-compose up -d        - Start Ollama server"
	@echo "  docker-compose logs -f      - View logs"
	@echo "  docker-compose down         - Stop server"
	@echo "  docker-compose down -v      - Stop and remove volumes"
