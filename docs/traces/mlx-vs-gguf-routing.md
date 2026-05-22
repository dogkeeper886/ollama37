# MLX vs GGUF: format detection & runner routing (upstream Ollama)

Traced against `ollama/ollama` upstream (via `gh api`) to confirm — from code, not docs —
what an `-mlx` tag actually changes and whether MLX is Apple-only. Our K80 fork does **not**
contain this code (only incidental `// matching MLX reference` comments in `model/models/gemma4`).

## Call flow

```
Scheduler picks runner by model format                 # server/sched.go:436
  ├── req.model.IsMLX() == false  → GGUF path
  │     ├── llm.LoadModel(ModelPath, 1024)              # sched.go:437
  │     └── s.newServerFn(...)                          # sched.go:443
  │           └── llama.cpp / GGML runner  ← the K80 fork patches THIS engine
  │
  └── req.model.IsMLX() == true   → MLX path           # sched.go:454
        ├── Config.Capabilities has "image"
        │     └── imagegen.NewServer(name)              # sched.go:456 (image GENERATION)
        └── else
              └── mlxrunner.NewClient(name)             # sched.go:458
                    └── exec "ollama runner --mlx-engine --model … --port …"
                                                         # x/mlxrunner/client.go:292
                          └── mlxrunner.Execute(args)    # runner/runner.go:22
```

## What decides "is this MLX?"

```go
func (m *Model) IsMLX() bool {                          // server/images.go:80
    return m.Config.ModelFormat == "safetensors"
}
```

The `-mlx` tag is **purely a manifest field**: `Config.ModelFormat == "safetensors"`.
- GGUF model → `ModelFormat != "safetensors"` → llama.cpp/GGML engine.
- MLX model → safetensors → MLX engine (`mlxrunner`), a separate subprocess (`--mlx-engine`).

Same weights, same architecture (`qwen3_5` underneath) — only the **storage format and the
runtime engine** differ.

## Is MLX Apple-only? NO.

```go
func CheckPlatformSupport() error {                     // x/imagegen/memory.go:26
    switch runtime.GOOS {
    case "darwin":   // requires arm64 (Apple Silicon)
    case "linux", "windows":  // "CUDA support (requires mlx or cuda build)"
        return nil            // backend availability checked at runtime
    ...
```

And the runner explicitly wires a **CUDA backend** on Linux:

```
x/mlxrunner/client.go:296-348
  ├── linux  → LD_LIBRARY_PATH += glob(mlx_*)
  ├── windows→ PATH
  └── glob(mlx_cuda_*) → set CUDA_PATH / CUDA_HOME
        # "Point MLX's JIT compiler at our bundled CUDA runtime headers.
        #  MLX resolves headers via $CUDA_PATH/include/*.h"
```

So MLX runs on Linux + NVIDIA via MLX's own CUDA backend (JIT-compiled kernels). My earlier
assertion "MLX can't run on K80 because it's Apple Silicon" was **wrong** — Apple Silicon is
only the *macOS* requirement.

## The real K80 question (UNVERIFIED)

MLX's CUDA backend JIT-compiles kernels against CUDA headers. Whether it supports
**compute capability 3.7 (sm_37, Kepler / K80)** is not yet confirmed from code. MLX-CUDA is
new (2025) and almost certainly targets modern archs (likely sm_70+ / Volta, or sm_50+ at
best). If it doesn't emit sm_37, MLX-on-K80 fails at JIT/launch — the *same* compute-cap wall
this fork already climbed for llama.cpp, but inside MLX's codebase.

## Bottom line

- `-mlx` = safetensors manifest → routes to the MLX engine instead of llama.cpp/GGML.
- MLX is **not** Apple-only; upstream has a Linux CUDA backend.
- To even attempt MLX on K80 we'd need to (a) port `x/mlxrunner` + MLX CUDA libs into this
  fork (absent today), and (b) confirm MLX-CUDA can target sm_37 — the decisive unknown.
- For the **base** models there's no payoff: `qwen3.6:27b`/`:35b` GGUF already run on K80 and
  are the same architecture. MLX only matters for models published **MLX-only** (e.g. the
  coding fine-tunes), and only if (a)+(b) hold.
