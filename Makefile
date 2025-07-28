# SpInfer Core Library Makefile
# Copyright 2025 The SpInfer Authors. All rights reserved.

# Compiler settings
CUDA_PATH ?= /usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc
HOST_COMPILER ?= g++

# Build directories
BUILD_DIR := build
OBJ_DIR := $(BUILD_DIR)/obj
LIB_DIR := $(BUILD_DIR)

# Source files
CSRC_DIR := csrc
CUDA_SOURCES := $(CSRC_DIR)/SpMM_API.cu

# Header files
HEADERS := $(CSRC_DIR)/SpMM_API.cuh \
           $(CSRC_DIR)/MatMulUtilities.cuh \
           $(CSRC_DIR)/SpMM_Kernel.cuh \
           $(CSRC_DIR)/TilingConfig.h \
           $(CSRC_DIR)/MMA_PTX.cuh \
           $(CSRC_DIR)/AsyncCopy_PTX.cuh \
           $(CSRC_DIR)/Reduction_Kernel.cuh

# Target library
TARGET_LIB := $(LIB_DIR)/libSpMM_API.so

# Compiler flags
NVCC_FLAGS := -std=c++14 -O3 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89
NVCC_FLAGS += -Xcompiler -fPIC -shared
NVCC_FLAGS += -DWITH_CUDA

# Include paths
INCLUDES := -I$(CSRC_DIR) -I$(CUDA_PATH)/include

# Libraries
LIBRARIES := -lcudart -lcublas -lcusparse

# Build rules
.PHONY: all clean install

all: $(TARGET_LIB)

$(TARGET_LIB): $(CUDA_SOURCES) $(HEADERS) | $(LIB_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $(CUDA_SOURCES) $(LIBRARIES)
	@echo "SpInfer library built successfully: $@"

$(LIB_DIR):
	@mkdir -p $(LIB_DIR)

clean:
	rm -rf $(BUILD_DIR)

install: $(TARGET_LIB)
	@echo "Installing SpInfer library..."
	@mkdir -p /usr/local/lib
	@mkdir -p /usr/local/include/spinfer
	@cp $(TARGET_LIB) /usr/local/lib/
	@cp $(CSRC_DIR)/SpMM_API.cuh /usr/local/include/spinfer/
	@echo "Installation completed"

# Help target
help:
	@echo "SpInfer Makefile Help:"
	@echo "  all     - Build the SpInfer library"
	@echo "  clean   - Remove build files"
	@echo "  install - Install library to system paths"
	@echo "  help    - Show this help message"