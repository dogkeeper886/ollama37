# Porting upstream code to the K80

This fork runs Ollama on the **Tesla K80** — compute capability **sm_37**, CUDA 11.4,
GCC 10, **no tensor cores**, 12 GB per die. Porting upstream code onto that hardware has
two disciplines. The first is about *getting it to build*; the second — the one most
easily skipped — is about *confirming it's actually faster*.

## 1. Port the source; don't adapt the environment

Vendor the upstream files into our tree and rewrite each one to compile with the
**existing** toolchain (nvcc 11.4 / GCC 10 / C++17 / sm_37, with cuBLAS or SIMT
fallbacks where tensor cores would be used). Mirror how `ml/backend/ggml` already vendors
and patches ggml in-tree.

Do **not** change the builder image, pull upstream source unmodified, or otherwise bend
the environment to the code. The fork exists to make code run on constrained hardware — if
using upstream unmodified were acceptable, upstream Ollama would already be the answer.

## 2. Benchmark every ported optimization on the K80 — it may be a net *loss*

Upstream optimizations are tuned for modern GPUs (tensor cores, large register files,
kernels written for Ampere/Hopper). On sm_37 an "optimization" can be **slower** than the
naive path it replaces. Building and producing correct output is **not** evidence that a
ported optimization helps — measure it.

**Worked example — flash attention.** Both upstream and our own `docker/.env.example`
assumed FA was a small win ("~3% tok/s"). On the K80 it is the opposite: for `gemma4:26b`,
FA off decodes ~22% faster at short context and **7.4× faster** at a 6,800-token prompt
(8.74 vs 1.18 tok/s; a 25-minute tool-calling run drops to under 3 minutes with FA off). The
"optimization" was the bottleneck. See **#337**. FA is now gated off on the K80 in code
(`FlashAttentionSupported` in `ml/device.go`, #347, #385).

**Rule:** when you port or enable an optimization — a flash-attention path, a fused kernel,
a tensor-core routine, a KV-cache quantization — benchmark it on the K80 against the path
it replaces, and keep it only if it actually wins. Don't trust upstream's performance
assumptions or in-code comments; they were written for different silicon.

## 3. Measure at realistic context lengths, not just short prompts

The FA regression was only +22% at a short prompt but **7.4×** at 6,800 tokens — a
short-prompt benchmark would have hidden it. Attention-related costs scale with context, so
profile at the context sizes real workloads use (large tool menus, long documents), not
just a one-line prompt.

## How to benchmark on the K80

The steps — which workflow, which inputs — live in the project's skills and
[`cicd/README.md`](../cicd/README.md), and change with CI. This section holds only the rules.

- **Study on the host first, then confirm through CI.** The host drives the K80 directly, so
  code changes, studies and lab runs start there; CI confirms what ships.
- **Check what else holds the GPUs before a long run.** Other agents share the host.
- **Compare against the path the change replaces**, on the same model, prompt and context
  size, run the same way.
- **Change one variable at a time.** When one setting forces another — `q8_0` KV cache
  requires FA, so FA off also means an `f16` cache — name the confound next to the numbers.
- **Report prefill and decode separately.** They stress different paths, and a change can
  win one and lose the other. [Reading prefill and decode speed in the server
  log](./server-log-metrics.md) covers where the log's numbers differ from the API's.
- **Restore every setting you changed** once the run is done.
