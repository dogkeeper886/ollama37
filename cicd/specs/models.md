# Models Tests

Model compatibility tests for ollama37 on Tesla K80 (CUDA compute 3.7).

---

## TC-MODELS-001: gpt-oss:20b Inference

**Importance:** High
**Execution Type:** Automated

**Summary:**
Validate gpt-oss:20b (~20B params, ~15GB VRAM) runs on K80.

### Steps

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Pull model:<br>`docker exec ollama37 ollama pull gpt-oss:20b` | Model downloads or already exists |
| 2 | Test inference via API:<br>`curl -s http://localhost:11434/api/generate -d '{"model":"gpt-oss:20b","prompt":"What is 2+2?","stream":false}'` | Returns JSON with 'response' field |
| 3 | Check GPU memory:<br>`docker exec ollama37 nvidia-smi --query-compute-apps=pid,used_memory --format=csv` | Shows GPU memory allocation |
| 4 | Unload model:<br>`curl -s http://localhost:11434/api/generate -d '{"model":"gpt-oss:20b","keep_alive":0}'` | Model unloaded from VRAM |

---

## TC-MODELS-002: gemma3:27b Inference

**Importance:** High
**Execution Type:** Automated

**Summary:**
Validate gemma3:27b (~27B params, ~18GB VRAM) runs on K80.

### Steps

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Pull model:<br>`docker exec ollama37 ollama pull gemma3:27b` | Model downloads or already exists |
| 2 | Test inference via API:<br>`curl -s http://localhost:11434/api/generate -d '{"model":"gemma3:27b","prompt":"What is 2+2?","stream":false}'` | Returns JSON with 'response' field |
| 3 | Check GPU memory:<br>`docker exec ollama37 nvidia-smi --query-compute-apps=pid,used_memory --format=csv` | Shows GPU memory allocation |
| 4 | Unload model:<br>`curl -s http://localhost:11434/api/generate -d '{"model":"gemma3:27b","keep_alive":0}'` | Model unloaded from VRAM |

---

## TC-MODELS-003: deepseek-r1:14b Inference

**Importance:** High
**Execution Type:** Automated

**Summary:**
Validate deepseek-r1:14b (~14B params, ~10GB VRAM) runs on K80.

### Steps

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Pull model:<br>`docker exec ollama37 ollama pull deepseek-r1:14b` | Model downloads or already exists |
| 2 | Test inference via API:<br>`curl -s http://localhost:11434/api/generate -d '{"model":"deepseek-r1:14b","prompt":"What is 2+2?","stream":false}'` | Returns JSON with 'response' field |
| 3 | Check GPU memory:<br>`docker exec ollama37 nvidia-smi --query-compute-apps=pid,used_memory --format=csv` | Shows GPU memory allocation |
| 4 | Unload model:<br>`curl -s http://localhost:11434/api/generate -d '{"model":"deepseek-r1:14b","keep_alive":0}'` | Model unloaded from VRAM |

---

## Notes

- **VRAM Constraint:** K80 has 22GB total (2 x 11GB). Models tested one at a time with unloading.
- **Acceptable warning:** "flash attention enabled but not supported by gpu" - K80 does not support flash attention
- **No CUDA/CUBLAS errors** should appear in logs

---

## TC-MODELS-018: ornith:9b Inference

**Importance:** High
**Execution Type:** Automated
**Status:** Red until the ornith renderer/parser port (#422) is in the deployed image.

**Summary:**
Validate ornith:9b (Qwen3.5-family, dense `qwen35`, thinking) runs on K80. Request pins
`"think": false` so the answer isn't buried in a `<think>` span (ornith forces thinking, #422).

---

## TC-MODELS-019: ornith:35b Inference

**Importance:** High
**Execution Type:** Automated
**Status:** Red until the ornith renderer/parser port (#422) is in the deployed image.

**Summary:**
Validate ornith:35b (Qwen3.5-family MoE, `qwen35moe`, thinking) runs on K80. Larger than one
die, so it may span both K80 dies (`GPU_COUNT_OK` when it genuinely needs them).

---

## TC-MODELS-021: lfm2.5-thinking:1.2b Inference

**Importance:** High
**Execution Type:** Automated
**Status:** KNOWN RED — GGUF fails to load with `missing tensor 'token_embd_norm.weight'`;
gates the vendored-llama.cpp LFM2 loader fix.

**Summary:**
Validate lfm2.5-thinking:1.2b (Liquid LFM2 thinking, ~0.7 GB) runs on K80, single die. Request
pins `"think": false`.
