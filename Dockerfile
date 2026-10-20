# Spike: FLUX.2 [klein] 4B on CPU with stable-diffusion.cpp
ARG UBUNTU_VERSION=24.04

FROM ubuntu:${UBUNTU_VERSION} AS build

RUN apt-get update \
  && apt-get install -y --no-install-recommends build-essential cmake git ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# stable-diffusion.cpp master of 2026-10-18
ARG SDCPP_COMMIT=b13a48219d5cdf408edcb93da6cd69b612ed03d3

WORKDIR /sd.cpp

RUN git init -q . \
  && git remote add origin https://github.com/leejet/stable-diffusion.cpp \
  && git fetch -q --depth 1 origin ${SDCPP_COMMIT} \
  && git checkout -q FETCH_HEAD \
  && git submodule update -q --init --depth 1 ggml

ARG TARGETARCH

# CPU only
RUN if [ "$TARGETARCH" = amd64 ]; then \
      CPU_ARGS="-DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON"; \
    else \
      CPU_ARGS="-DGGML_NATIVE=ON"; \
    fi \
  && cmake -B build $CPU_ARGS \
    -DCMAKE_BUILD_TYPE=Release \
    -DSD_SERVER_BUILD_FRONTEND=OFF \
    -DSD_BUILD_SHARED_LIBS=ON \
    -DSD_BUILD_SHARED_GGML_LIB=ON \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DCMAKE_INSTALL_RPATH='$ORIGIN' \
  && cmake --build build --parallel "$(nproc)"

FROM ubuntu:${UBUNTU_VERSION}

RUN apt-get update \
  && apt-get install -y --no-install-recommends libgomp1 python3 ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY --from=build /sd.cpp/build/bin /opt/sd.cpp

# uid 1000, taken by the base image's "ubuntu" user
RUN userdel -r ubuntu && useradd -m -u 1000 huggingface

USER huggingface

WORKDIR /home/huggingface

RUN mkdir -p /home/huggingface/.cache/huggingface

COPY server.py /usr/local/bin

EXPOSE 8081

ENTRYPOINT [ "python3", "/usr/local/bin/server.py" ]
