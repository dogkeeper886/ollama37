# Inference Tests


---

## TC-INFERENCE-001: Model Pull

**Importance:** High
**Execution Type:** Automated

**Summary:**
Pull the gemma3:4b model for testing inside the container.

### Steps

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Pull test model:<br>`docker exec ollama37 ollama pull gemma3:4b` | Model downloads successfully or reports already exists |
| 2 | Verify model available:<br>`docker exec ollama37 ollama list` | gemma3:4b listed in output |

---

## TC-INFERENCE-002: API Inference Test

**Importance:** High
**Execution Type:** Automated

**Summary:**
Test inference via Ollama REST API /api/generate endpoint.

### Steps

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Test generate endpoint:<br>`curl -s http://localhost:11434/api/generate -d '{"model":"gemma3:4b","prompt":"What is 2+2? Answer with just the number.","stream":false}'` | Returns JSON with 'response' field |
| 2 | Check GPU memory usage:<br>`docker exec ollama37 nvidia-smi --query-compute-apps=pid,used_memory --format=csv` | Ollama process shows GPU memory allocation |
| 3 | Unload model after test:<br>`curl -s http://localhost:11434/api/generate -d '{"model":"gemma3:4b","keep_alive":0}'` | Model unloaded from VRAM |

### Notes

- **Acceptable warning:** "flash attention enabled but not supported by gpu" - K80 does not support flash attention, this is a normal fallback warning, NOT an error
- No CUDA/CUBLAS errors should appear in output
