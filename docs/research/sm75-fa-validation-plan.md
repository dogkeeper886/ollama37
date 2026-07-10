# sm_75 (RTX 2060) validation — flash-attention plan

Working plan for the non-K80 hardware validation in **#342** (plan **#340**, cards **#358**).
Also serves the *Done When* item "a runnable smoke-test procedure is documented".

**Status:** T1, T3, T4, T5, T6 executed. T2 blocked (model unsupported). See *Results*.
**Last updated:** 2026-07-10.
**Thread:** <https://github.com/dogkeeper886/ollama37/issues/342#issuecomment-4934285756>

## Results (2026-07-10)

| # | Config | Engine | FA | Observed | Verdict |
|--:|---|---|---|---|---|
| T1 | `deepseek-r1:1.5b`, defaults | llama.cpp | off | 153.9 tok/s, `"Paris"`, `done=stop` | **PASS** |
| T2 | `lfm2.5-thinking:1.2b`, defaults | llama.cpp | — | `error loading model: missing tensor 'token_embd_norm.weight'`, `print_info: arch = lfm2` | **BLOCKED** |
| T3 | `qwen3.5:0.8b`, **no `.env`** | **Go** | **true (auto)** | `architecture=qwen35`, 25/25 on GPU, then `panic: failed to sample token` (`ollamarunner/runner.go:737`) | **CRASH** |
| T4 | `qwen3.5:0.8b`, `FA=0` | Go | false | 125.4 tok/s, coherent reasoning to "Paris", 0 panics | **PASS** |
| T5 | `deepseek-r1:1.5b`, `FA=1` | llama.cpp | true | `Assertion 'found' failed` + `SIGABRT` | **CRASH** |
| T6 | `deepseek-r1:1.5b`, `FA=1`, `CUDA_VISIBLE_DEVICES=-1` | llama.cpp | true (`"enabling flash attention"`) | `library=cpu`, 4.77 tok/s, 0 aborts | **PASS** |

**What this establishes.**

1. **The default config is broken on cc>=7.0.** T3 set no environment variable at all and the
   load request still carried `FlashAttention:true`. The §2 whitelist is confirmed empirically,
   not just by reading source.
2. **The break is not engine- or arch-specific.** Two different samplers fail on the same kernel:
   llama.cpp's C++ `llama_sampler_dist_apply` aborts (T5, arch `qwen2`), and the Go engine's
   sampler panics (T3, arch `qwen35`). Different code, same cause.
3. **The fault is in the CUDA FA path.** T6 was designed to refute that: same model, same
   `FA=1`, no GPU. It passed. FA on CPU is fine. The `MMA_F16` / `movmatrix` attribution
   survived its falsification test.
4. **FA is the only variable.** T3 vs T4 hold model, engine and host constant; only FA differs.

**Still not proven:** that the `movmatrix` fallback is the *specific* defect. T6 narrows the
fault to the CUDA FA path; it does not single out `get_transposed` within it. Confirming that
still needs the builder-image probe or a CUDA 11.8 sm_75 build.

**New, unrelated finding (T2).** `lfm2.5-thinking:1.2b` is `arch = lfm2` (dense) and fails to
load: `missing tensor 'token_embd_norm.weight'`. Only `lfm2moe` (the 8b) was merged in #383.
This is a load-time architecture gap, not sm_75-specific — it would fail on the K80 too.
Deserves its own issue.

---

## The finding in one paragraph

The published image runs correctly on the RTX 2060 (sm_75) with flash attention **off**:
29/29 layers on GPU, correct output, 153.9 tok/s vs the K80's 34.1 on `deepseek-r1:1.5b`.
With FA **on** it aborts deterministically (2/2) in `llama_sampler_dist_apply` — the signature
of NaN logits. And **FA turns on by itself** for nine architectures, so the *default* config is
unsafe on any cc>=7.0 card, not just when a user follows `.env.example`.

## What is proven vs. suspected

**Proven** (source-traced + observed):

| Claim | Evidence |
|---|---|
| Env unset ⇒ FA falls back to a per-arch whitelist | `envconfig/config.go:152-163` (`BoolWithDefault`), `llm/server.go:244` |
| The whitelist | `fs/ggml/ggml.go:904` — gemma3, gptoss/gpt-oss, qwen3, qwen3moe, qwen35, qwen35moe, qwen3next, qwen3vl, qwen3vlmoe |
| cc>=7 passes the GPU gate | `ml/device.go:436` |
| `qwen35` bypasses the model gate | `fs/ggml/ggml.go:889` returns true unconditionally |
| Both engines hit the same CUDA op | llama.cpp via `llamarunner`; Go via `ml/backend/ggml/ggml.go:1798` → `C.ggml_flash_attn_ext` |
| sm_75 selects the tensor-core kernel | `fattn.cu:195` → `turing_mma_available(750)` true, VEC needs cc>=890 → `BEST_FATTN_KERNEL_MMA_F16` (`fattn.cu:295`) |
| CUDA 11.4 compiles the fallback `movmatrix` | `mma.cuh:22` gates on `CUDART_VERSION >= 11080`; image bundles `libcudart.so.11.4.148` (11040) |
| That fallback is reachable only via FA | `ggml_cuda_movmatrix` → `get_transposed` (`mma.cuh:213`) → called only at `fattn-mma-f16.cuh:696,1142` |
| K80 never executes it | `turing_mma_available(370)` is false |
| FA=1 aborts, FA=0 clean | 2/2 each, this host |

**Suspected — do not state as fact:**

- That the `movmatrix` fallback is *numerically wrong*. It is upstream code. The fault may lie
  elsewhere in `fattn-mma-f16` under nvcc 11.4. **T6 is the test that can refute this.**
- That `__CUDA_ARCH_LIST__` (`common.cuh:133`) is CUDA 11.5+. If undefined on 11.4,
  `ggml_cuda_highest_compiled_arch` degrades to `return arch` (`common.cuh:167`) and
  `turing_mma_available()` cannot detect a missing cubin. Confirm inside the builder.
- All arch strings in the matrix below. Confirm from `general.architecture` at load.

## Host + current state

| | |
|---|---|
| GPU | RTX 2060 (TU106), cc **7.5**, 6144 MiB, ~5.1 GiB free (display attached) |
| Driver | 580.159.03 (CUDA 13.0) — **not** the 470 target |
| OS / Docker | Rocky 9.8 · Docker 29.6.1 · Compose v5.3.1 |
| Image | `dogkeeper886/ollama37:latest` `sha256:32010cd3…`, self-reports `2.1.0`, retagged `ollama37:latest` |

Already done, do not redo:

- `nvidia-container-toolkit` installed; Docker `nvidia` runtime registered.
- `docker volume create ollama-data` (compose declares it `external: true`).
- Models in the volume: `deepseek-r1:1.5b`, `qwen3:4b` (**unwanted** — pulled by an aborted
  command; remove with `docker exec ollama37 ollama rm qwen3:4b`).
- `docker/.env` exists with `OLLAMA_FLASH_ATTENTION=0` (gitignored). **T2 and T3 require it
  deleted, not edited** — compose maps unset → empty string → `BoolWithDefault` default path.

## Model → path map on sm_75

Sizes are published tags; `available` on this host is 5.1 GiB.

| Model | Size | arch (expected) | Engine | FA, no env | Kernel | Fits | Expected |
|---|--:|---|---|---|---|:--:|---|
| `deepseek-r1:1.5b` | 1.1 GB | `qwen2` | llama.cpp | off | — | yes | PASS (observed) |
| `lfm2.5-thinking:1.2b` | 731 MB | `lfm2`/`lfm2moe` | llama.cpp | off | — | yes | PASS |
| `lfm2.5:8b` | 5.2 GB | `lfm2moe` | llama.cpp | off | — | **no** (K80 d0 6358 MiB) | untestable |
| `qwen3.5:0.8b` | 1.0 GB | `qwen35` | **Go** | **ON** | `MMA_F16` | yes | **CRASH** |
| `qwen3.5:2b` | 2.7 GB | `qwen35` | **Go** | **ON** | `MMA_F16` | yes | **CRASH** |
| `qwen3.5:4b` | 3.4 GB | `qwen35` | **Go** | **ON** | `MMA_F16` | yes | **CRASH** |
| `qwen3.5:latest` | 6.6 GB | `qwen35moe` | **Go** | **ON** | `MMA_F16` | **no** | untestable |

VRAM ceiling is from `docs/reports/k80-vram-by-family-2026-07-10-d83061de.md` (per-die `d0`).
`qwen35` is hybrid: `FullAttention` layers use FA (`attention.go:91`), `GatedDeltaNet` layers
bypass it (`deltanet.go`). `model/models/qwen35/` has **no vision tower**, yet the published
`qwen3.5:*` tags advertise Image — separate gap, needs its own issue.

## The matrix

| # | Model | Env | Isolates | Expect | Done |
|--:|---|---|---|---|:--:|
| T1 | `deepseek-r1:1.5b` | defaults | sm_75 cubin runs at all | PASS | ✅ |
| T2 | `lfm2.5-thinking:1.2b` | **no `.env`** | 2nd arch on the no-FA path | PASS | ☐ |
| T3 | `qwen3.5:0.8b` | **no `.env`** | **default config is broken** | CRASH | ☐ |
| T4 | `qwen3.5:0.8b` | `FA=0` | controls engine; isolates FA | PASS | ☐ |
| T5 | `lfm2.5-thinking:1.2b` | `FA=1` | FA break is engine/arch-independent | CRASH | ☐ |
| T6 | `deepseek-r1:1.5b` | `FA=1`, CPU only | **refutes the CUDA attribution** | PASS | ☐ |

T3 is the headline — no configuration at all. T6 is the falsifier: if CPU also aborts, the
`movmatrix` trace is wrong and the whole mechanism section must be retracted.

Optional: **T7** = T3 + `OLLAMA_KV_CACHE_TYPE=q8_0` (quantized K ⇒ decode takes `VEC`, prefill
stays `MMA_F16` — the only way to separate the two kernels without a rebuild).
**T8** = image prompt to `qwen3.5:4b`, documents the missing vision tower.

## Procedure

Pull first (~1.8 GB total): `docker exec ollama37 ollama pull qwen3.5:0.8b` and
`... pull lfm2.5-thinking:1.2b`.

**No-`.env` runs (T2, T3):**

```bash
cd docker && mv .env .env.bak && docker compose up -d --force-recreate && sleep 6
docker logs ollama37 2>&1 | grep -o "OLLAMA_FLASH_ATTENTION:[a-z]*" | tail -1   # expect: false (empty env)
```

**`.env` runs (T4, T5):** set `OLLAMA_FLASH_ATTENTION=0` or `=1`, then
`docker compose up -d --force-recreate`.

**T6 (CPU control):**

```bash
docker run -d --rm --name ollama37-cpu -e OLLAMA_FLASH_ATTENTION=1 \
  -e CUDA_VISIBLE_DEVICES=-1 -e OLLAMA_HOST=0.0.0.0:11434 \
  -p 11435:11434 -v ollama-data:/root/.ollama ollama37:latest
# ... then curl :11435 ; docker stop ollama37-cpu when done
```

**Drive one model** (mirrors the K80 sweep methodology: `num_predict=16`, 4k ctx):

```bash
curl -s --max-time 240 http://localhost:11434/api/generate -d '{
  "model":"<MODEL>","prompt":"In one sentence, what is the capital of France?",
  "stream":false,"options":{"num_predict":16,"num_ctx":4096,"seed":42}}' > out.json
python3 -c 'import json;d=json.load(open("out.json"));ec,ed=d.get("eval_count"),d.get("eval_duration");print("eval:",ec,"| tok/s:", round(ec/(ed/1e9),2) if ec and ed else None, "| error:",d.get("error","none"))'
```

**Record per run** — an empty `eval_count` plus a 500 means the runner died:

1. `general.architecture` — `docker logs ollama37 2>&1 | grep -i "general.architecture\|architecture "`
2. FA in effect — `grep -o "OLLAMA_FLASH_ATTENTION:[a-z]*"`, and `FlashAttention:true|false` in the load request
3. Engine — `llama_model_load_from_file_impl` (llama.cpp) vs the Go runner's load lines
4. GPU placement — `layers.offload=N` and `available=`
5. tok/s, output text, `done_reason`
6. On crash — `grep -c "Assertion \`found' failed"` and the `SIGABRT` block

Known limitation: **the selected FA kernel is not logged.** MMA-vs-VEC is inferred from
`fattn.cu:195`, not observed. T7 is the only cheap discriminator.

## Cleanup / restore

```bash
cd docker && cp .env.bak .env 2>/dev/null || printf 'OLLAMA_FLASH_ATTENTION=0\n' > .env
docker compose up -d --force-recreate
docker stop ollama37-cpu 2>/dev/null
docker exec ollama37 ollama rm qwen3:4b     # unwanted
```

Leave the box at `FA=0` — the default config crashes for whitelisted archs.

## Decisions pending

- **Doc/gate fix — needs no further testing.** The default auto-enables FA on every cc>=7.0
  card for nine archs. Either hard-gate FA off in this build, or fix `docker-compose.yml:22-23`
  (which claims "we don't auto-enable it on the experimental cc>=7.0 cards" — false) and
  `.env.example:17-20`. Amends **#340 step 2**, which plans to leave the gate untouched on the
  belief the path is "pure-upstream and already correct" for every card in the sweep.
- **Deferred, expensive:** `get_transposed` probe in the builder (~15 GB image); an sm_75-only
  CUDA 11.8 build. Both buy root cause for a feature the K80 does not want. Revisit only if a
  Pascal card or the #223 reporter appears.
- **New issue:** `qwen3.5:*` advertise Image but `model/models/qwen35/` has no vision tower.
- `docs/research/470-arch-support-map.md` is referenced by #340 and #342 but was deleted in
  `17678ce4`. This file is the "or a sibling" it points to.

## Out of scope, by construction

Pascal gate crossings (sm_60 fast-FP16, sm_61 dp4a), the #223 mixed-arch box, driver 470,
and any model above 5.1 GiB. This host is one Turing card on a modern driver — a weak
best-effort sanity check, exactly as #342 frames it.
