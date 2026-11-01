# Ollama37 Builder Image
# This image provides the complete build environment for compiling Ollama with Tesla K80 (compute 3.7) support
# Includes: CUDA 11.4, GCC 10, CMake 4, Go 1.26.10

FROM rockylinux/rockylinux:8

# Install CUDA toolkit 11.4
# Note: NVIDIA driver is NOT needed in builder container - only CUDA toolkit for compilation
# The host system provides the driver at runtime via --gpus flag
# dnf-plugins-core is required for the config-manager command

RUN dnf -y install dnf-plugins-core\
    && dnf -y config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo\
    && dnf -y install cuda-toolkit-11-4

# Setup CUDA path
RUN echo 'export PATH="${PATH}:/usr/local/cuda-11.4/bin"' > /etc/profile.d/cuda-11.4.sh
ENV PATH="$PATH:/usr/local/cuda-11.4/bin"

# Install GCC 10 from source
# CUDA 11.4 requires GCC 10 maximum (enforced in host_config.h)
# GCC 11+ is incompatible with CUDA 11.4
RUN dnf -y install wget unzip bzip2 git\
    && dnf -y groupinstall "Development Tools"\
    && cd /usr/local/src\
    && wget https://github.com/gcc-mirror/gcc/archive/refs/heads/releases/gcc-10.zip\
    && unzip gcc-10.zip\
    && cd gcc-releases-gcc-10\
    && contrib/download_prerequisites\
    && mkdir /usr/local/gcc-10\
    && cd /usr/local/gcc-10\
    && /usr/local/src/gcc-releases-gcc-10/configure --disable-multilib\
    && make -j $(nproc)\
    && make install

# Setup GCC 10 library path and update system compiler
# Configure ldconfig to find GCC 10 runtime libraries
# Replace default cc symlink to use our custom GCC 10
RUN echo '/usr/local/lib64' > /etc/ld.so.conf.d/gcc-10.conf\
    && ldconfig\
    && rm -f /usr/bin/cc\
    && ln -s /usr/local/bin/gcc /usr/bin/cc

# Install CMake 4 from source
# Required for modern CMake features and CUDA architecture configuration
RUN dnf -y install openssl-devel\
    && cd /usr/local/src\
    && wget https://github.com/Kitware/CMake/releases/download/v4.0.0/cmake-4.0.0.tar.gz\
    && tar xvf cmake-4.0.0.tar.gz\
    && mkdir /usr/local/cmake-4\
    && cd /usr/local/cmake-4\
    && /usr/local/src/cmake-4.0.0/configure\
    && make -j $(nproc)\
    && make install

# Install go 1.26.10
RUN cd /usr/local\
    && wget https://go.dev/dl/go1.26.10.linux-amd64.tar.gz\
    && tar xvf go1.26.10.linux-amd64.tar.gz

# FetchContent / hermetic build policy:
# llama/server/CMakeLists.txt fetches llama.cpp via FetchContent at configure
# time, pinned to the immutable GIT_TAG read from LLAMA_CPP_VERSION (b11963).
# The tag is immutable so CI cannot silently drift.  Offline builds must
# pre-seed the source: set OLLAMA_LLAMA_CPP_SOURCE to a local llama.cpp tree,
# or set FETCHCONTENT_SOURCE_DIR_LLAMA_CPP directly — OLLAMA_LLAMA_CPP_SOURCE
# is honoured by llama/server (see its CMakeLists.txt).

# Setup Go path
RUN echo 'export PATH="${PATH}:/usr/local/go/bin"' > /etc/profile.d/go.sh
ENV PATH="$PATH:/usr/local/go/bin"
