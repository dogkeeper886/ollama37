# Ollama37 Profile Image
# Adds nvprof (CUDA 11.4 profiler) to the runtime image for K80 kernel profiling.
# Used by .github/workflows/test-profile.yml — not for production.
#
# Size delta: ~3-4 GB on top of ollama37:latest (full CUDA 11.4 toolkit).
# Targeted copies of just nvprof + CUPTI proved insufficient — nvprof loads
# additional injection libraries at runtime (libaccinj64, libcuinj64, etc.)
# whose paths are baked into nvprof's binary. Easier to ship the whole toolkit
# than to reverse-engineer the dependency closure.

FROM ollama37-builder AS nvprof_source

FROM ollama37:latest

# Copy the entire CUDA 11.4 toolkit from the builder image. Includes nvprof,
# all profiling injection libraries (libaccinj64, libcuinj64), CUPTI, and
# the rest. Image size is irrelevant for a one-off profiling tool.
COPY --from=nvprof_source /usr/local/cuda-11.4 /usr/local/cuda-11.4

ENV PATH="$PATH:/usr/local/cuda-11.4/bin"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-11.4/lib64:/usr/local/cuda-11.4/extras/CUPTI/lib64"

# Override entrypoint so the workflow can wrap ollama with nvprof
ENTRYPOINT []
CMD ["sh", "-c", "echo 'Use docker run with explicit nvprof invocation. See .github/workflows/test-profile.yml.'"]
