# Ollama37 🚀

**Tesla K80 Compatible Ollama Fork**

Run modern LLMs on NVIDIA Tesla K80 and other CUDA Compute Capability 3.7 GPUs. While official Ollama dropped legacy GPU support, Ollama37 keeps your Tesla K80 hardware functional with the latest models and features.

## Key Features

- ⚡ **Tesla K80 Support** - Full compatibility with CUDA Compute Capability 3.7
- 🛠️ **Optimized Build** - CUDA 11 toolchain with native CUBIN (~30s cold start)
- 🧠 **Qwen3.5 DeltaNet** - First Ollama fork with DeltaNet recurrent architecture support
- 🔧 **Tool Calling** - Langchain and API tool calling verified

## Tested Models

| Model | Size | Status |
|-------|------|--------|
| gemma4:e4b | 4B | ✅ |
| gemma4:26b | 26B | ✅ |
| gemma3:4b | 4B | ✅ |
| gemma3:27b | 27B | ✅ |
| qwen3.5:9b | 9B | ✅ |
| qwen3.5:27b | 27B | ✅ |
| gpt-oss:20b | 20B | ✅ |
| ministral-3 | 3B | ✅ |
| functiongemma | 270M | ✅ |
| deepseek-r1:7b | 7B | ✅ |

🎥 **Demo video:** [Ollama37 v2.0.3 on Tesla K80](https://youtu.be/B1PbLr3rUhc)

## Quick Start

### Docker (Recommended)
```bash
# Pull and run
docker pull dogkeeper886/ollama37:v2.1.0
docker run --runtime=nvidia --gpus all -p 11434:11434 dogkeeper886/ollama37
```

### Docker Compose
```yaml
version: "3.8"

services:
  ollama:
    image: dogkeeper886/ollama37:latest
    container_name: ollama37
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    ports:
      - "11434:11434"
    volumes:
      - ollama-data:/root/.ollama
    environment:
      - OLLAMA_HOST=0.0.0.0:11434
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "/usr/local/bin/ollama", "list"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 5s

volumes:
  ollama-data:
    name: ollama-data
```
```bash
docker-compose up -d
```

## Usage

### Run Your First Model
```bash
# Download and run a model
ollama pull gemma3
ollama run gemma3 "Why is the sky blue?"

# Interactive chat
ollama run gemma3
```

### CLI Commands
```shell
ollama list              # List models
ollama show llama3.2     # Model info
ollama ps               # Running models
ollama stop llama3.2    # Stop model
ollama serve            # Start server
```

## Contributing

Found an issue or want to contribute? Check our [GitHub issues](https://github.com/dogkeeper886/ollama37/issues) or submit Tesla K80-specific bug reports and compatibility fixes.

## License

Same license as upstream Ollama. See LICENSE file for details.
