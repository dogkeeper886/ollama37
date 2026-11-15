# ollama37 K80 (Kepler, compute 3.7) compatibility patch series.
#
# Ordered *.patch set applied to the fetched llama.cpp AFTER the upstream
# llama/compat patches, against the pinned LLAMA_CPP_VERSION ref. Same
# idempotent applier as llama/compat (cmake/apply-git-patches.cmake).
#
# Include from llama/server/CMakeLists.txt right after compat.cmake; the
# resulting command is chained into FetchContent's PATCH_COMMAND.

set(_compat_k80_dir ${CMAKE_CURRENT_LIST_DIR})
get_filename_component(_ollama_k80_patch_applier
    "${_compat_k80_dir}/../../cmake/apply-git-patches.cmake"
    ABSOLUTE)

set(OLLAMA_LLAMA_CPP_COMPAT_K80_PATCH_COMMAND
    ${CMAKE_COMMAND}
        -DPATCH_DIR=${_compat_k80_dir}
        -DPATCH_LABEL=llama/compat-k80
        -P ${_ollama_k80_patch_applier}
    CACHE INTERNAL "llama.cpp compat-k80 patch command for FetchContent")
