# Runtime Tests


---

## TC-RUNTIME-001: Container Startup

**Importance:** High
**Execution Type:** Automated

**Summary:**
Start the ollama37 container with GPU passthrough using docker compose.

### Steps

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Navigate to docker directory:<br>`cd /home/jack/src/ollama37/docker` | Directory exists |
| 2 | Start container:<br>`docker compose up -d` | Container starts without errors |
| 3 | Check container status:<br>`docker compose ps` | Container status: Up, healthy |

---

## TC-RUNTIME-002: GPU Detection

**Importance:** High
**Execution Type:** Automated

**Summary:**
Verify Tesla K80 GPU is detected by both nvidia-smi AND Ollama CUDA runtime. Includes UVM device workaround.

### Steps

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Check nvidia-smi inside container:<br>`docker exec ollama37 nvidia-smi` | Output shows Tesla K80 GPU(s) with Driver 470.x, CUDA 11.4 |
| 2 | Check CUDA libraries:<br>`docker exec ollama37 ldconfig -p \| grep -i cuda \| head -5` | CUDA libraries listed (libnvrtc, libcublas, etc.) |
| 3 | Check UVM device files and create if missing:<br>`ls /dev/nvidia-uvm \|\| sudo nvidia-modprobe -u -c=0`<br>If created, restart container:<br>`docker compose restart` | /dev/nvidia-uvm and /dev/nvidia-uvm-tools exist. Container restarts if devices were created. |
| 4 | Check Ollama GPU detection in logs:<br>`docker compose logs \| grep "inference compute" \| tail -5` | Log shows: library=CUDA compute=3.7 description=Tesla K80<br>NOT: library=cpu |

---

## TC-RUNTIME-003: Health Check

**Importance:** High
**Execution Type:** Automated

**Summary:**
Verify Ollama server health check passes and API is responsive.

### Steps

| # | Action | Expected Result |
|---|--------|-----------------|
| 1 | Check container health status:<br>`docker inspect ollama37 --format='{{.State.Health.Status}}'` | Output: healthy |
| 2 | Test API endpoint:<br>`curl -s http://localhost:11434/api/tags` | Returns JSON response (empty models list or existing models) |
| 3 | Check Ollama version:<br>`docker exec ollama37 ollama --version` | Version information displayed |
