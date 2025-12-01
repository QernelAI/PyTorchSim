# syntax=docker/dockerfile:1.4
FROM ghcr.io/psal-postech/torchsim_base:latest

# Prepare PyTorchSim project
COPY . /workspace/PyTorchSim

RUN cd PyTorchSim/TOGSim && \
    mkdir -p build && \
    cd build && \
    conan install .. --build=missing && \
    cmake .. && \
    make -j$(nproc)