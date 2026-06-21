# STORY-015: K80 inference stops paying the flash-attention penalty

## User Story

As a maintainer running models on the Tesla K80,
I want to remove the K80 flash-attention special-case so the K80 falls back to upstream's (FA-off) behavior,
So that long-context and tool-calling runs aren't ~7× slower than they need to be on hardware where flash attention is a net loss.

## The Need

`ml/device.go` `FlashAttentionSupported()` carries a K80-specific clause that force-enables flash attention on compute 3.7. It was added in #108 (2026-04) to unlock the q8_0 KV cache (~47% KV memory saving) and was validated for **correctness** — bit-exact against the non-FA path on gemma3:4b. Its **speed** cost was never measured.

A benchmark this session (#337) measured it on `gemma4:26b`: flash attention on the K80 is a **~7.4× slowdown at long context** (1.18 vs 8.74 tok/s decode on a 6,800-token prompt — a 25-minute tool-calling run drops to under 3 minutes) and ~29% slower on short prompts. On sm_37 (no tensor cores) the FA kernel is simply a net loss, and the special-case has been quietly imposing that cost on every K80 run.

Removing the clause reverts the gate to pure upstream (`cc >= 7.0`, ≠ 7.2): K80 and Pascal get FA **off**, V100/T4/A100 get FA **on**. That is also exactly what the STORY-014 sync point (#345) is waiting for — STORY-014 deliberately leaves `ml/device.go` to this separate effort so the two don't collide.

The cost is real and must be handled, not hidden: with FA off the q8_0 KV cache is no longer usable (V-cache quantization requires FA), so the K80 default cache must move to f16 — roughly **2× the KV memory**. KV rotation (#102) only benefits *quantized* KV, so on f16 it becomes a no-op carrying ~2–5% overhead.

## Success Looks Like

- K80 long-context decode runs at the ~7× faster non-FA speed — the FA penalty is gone for the workloads (tool-calling, long prompts) where it hurt most.
- `FlashAttentionSupported()` is pure-upstream with no K80 special-case, so the gate is correct across the widened arch sweep — unblocking STORY-014 (#345).
- The q8_0 → f16 VRAM tradeoff is **validated and documented** (does f16 still fit the largest supported models at long context?), not assumed — a maintainer knows what the change costs in memory before it ships.
- A run that was correct before is still correct (FA-off output already validated in the #337 benchmark; no quality regression).

## Open Questions

- Whether to flip the docker default KV cache to f16 unconditionally for the K80 deployment, or gate it — the **VRAM-fit validation on the largest models at long context is the key risk** and decides this.
- Whether and how to skip the now-pointless rotation on the f16 path (a ~2–5% optimization, possibly deferred to a follow-up).
- Merge-order coordination with STORY-014 on `ml/device.go` so the two efforts don't conflict (per #345 — post the diff/merge there to unblock).

## Status

- Created: 2026-06-21
- Evidence: #337 (benchmark + decision), #345 (STORY-014 sync point)
- Plan: #346
- Issues: #347 (code change), #348 (f16 VRAM validation), #349 (rotation-skip follow-up)
