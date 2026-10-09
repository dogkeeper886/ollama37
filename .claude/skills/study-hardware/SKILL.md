---
description: >-
  Code runs fastest when its hot path uses what the hardware does well. An
  agent picks what to rewrite by its own ideas: it reads the source and guesses
  at the slow part. The guess ignores how the hardware executes, what the build
  chain can compile, and which code path actually runs on that device, so the
  rewrite lands on the wrong part or never builds. This skill asks the agent to
  study the hardware, the build environment and the code path by the steps
  below.
---

# Study the hardware, its build environment and the code path before a rewrite

1. Read the host's GPUs, memory, clocks and build chain (GPU toolkit and compiler), then trace the code path.
2. Find what the hardware is good at: look up its per-unit instruction rates in the vendor's programming guide, and take the highest rates as its strengths and the lowest or missing ones as its weaknesses.
3. Measure each device's read ceiling with a plain-read kernel, and note any device that runs throttled.
4. Mark the part of the code path that leans on what the hardware lacks or does slowly as the part to rewrite.
