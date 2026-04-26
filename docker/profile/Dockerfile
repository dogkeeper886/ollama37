# Ollama37 Profile Image
# Adds nvprof (CUDA 11.4 profiler) to the runtime image for K80 kernel profiling.
# Used by .github/workflows/test-profile.yml — not for production.
#
# Size delta: ~200 MB on top of ollama37:latest (nvprof + CUPTI + deps).

FROM ollama37-builder AS nvprof_source

FROM ollama37:latest

# Copy nvprof and CUPTI from the builder image (which already has cuda-toolkit-11-4)
# Avoids the need to install dnf + the NVIDIA repo into the minimal runtime image.
COPY --from=nvprof_source /usr/local/cuda-11.4/bin/nvprof /usr/local/cuda-11.4/bin/nvprof
COPY --from=nvprof_source /usr/local/cuda-11.4/extras/CUPTI /usr/local/cuda-11.4/extras/CUPTI
COPY --from=nvprof_source /usr/local/cuda-11.4/lib64/libcupti.so* /usr/lib64/
COPY --from=nvprof_source /usr/local/cuda-11.4/lib64/libcupti_static.a /usr/lib64/

ENV PATH="$PATH:/usr/local/cuda-11.4/bin"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-11.4/extras/CUPTI/lib64"

# Override entrypoint so the workflow can wrap ollama with nvprof
ENTRYPOINT []
CMD ["sh", "-c", "echo 'Use docker run with explicit nvprof invocation. See .github/workflows/test-profile.yml.'"]
