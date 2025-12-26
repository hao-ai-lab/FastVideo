cmake_minimum_required(VERSION 3.26 FATAL_ERROR)
project(fastvideo-kernel LANGUAGES CXX CUDA)

# Import common utils if needed, but we keep it simple for now

# Find Python and Torch
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Robustly find Torch include paths using Python
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import torch; from torch.utils.cpp_extension import include_paths; print(';'.join(include_paths()))"
    OUTPUT_VARIABLE TORCH_INCLUDE_PATHS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND TORCH_INCLUDE_DIRS ${TORCH_INCLUDE_PATHS})

# Find Torch package (still useful for libraries)
find_package(Torch REQUIRED)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/tk/include
    ${CMAKE_SOURCE_DIR}/include/tk/prototype
    ${CMAKE_SOURCE_DIR}/csrc
    ${TORCH_INCLUDE_DIRS}
)

# CUDA architecture flags
# Force usage of only SM90a because ThunderKittens depends on it.
set(TORCH_CUDA_ARCH_LIST "9.0a" CACHE STRING "Force SM90a for ThunderKittens" FORCE)
set(CMAKE_CUDA_ARCHITECTURES "90a" CACHE STRING "Force SM90a" FORCE)

# Compiler flags
set(CUDA_FLAGS
    "-DNDEBUG"
    "-O3"
    "-std=c++20"
    "--use_fast_math"
    "--expt-extended-lambda"
    "--expt-relaxed-constexpr"
    "-Xcompiler=-fno-strict-aliasing"
    "-Xcompiler=-fPIC"
    "-DTORCH_COMPILE"
    "-DKITTENS_HOPPER"
    "-Xnvlink=--verbose"
    "-Xptxas=--verbose"
    "-Xptxas=--warn-on-spills"
)

# Function to get nvcc flags for target architecture
# Here we just hardcode sm_90a since it's TK requirement generally
list(APPEND CUDA_FLAGS "-arch=sm_90a")

    # Combined FastVideo Extension
    # Using name 'fastvideo_kernel_ops' to distinguish from the python package namespace
    Python_add_library(fastvideo_kernel_ops MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI
        csrc/common_extension.cpp
        csrc/attention/st_attn_h100.cu
        csrc/attention/block_sparse_h100.cu
    )
    
    target_compile_options(fastvideo_kernel_ops PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS} -DTK_COMPILE_ST_ATTN -DTK_COMPILE_BLOCK_SPARSE>
        $<$<COMPILE_LANGUAGE:CXX>:-DTK_COMPILE_ST_ATTN -DTK_COMPILE_BLOCK_SPARSE>
    )
    
    # We install it to fastvideo_kernel/_C so we can load it to register the ops
    install(TARGETS fastvideo_kernel_ops LIBRARY DESTINATION fastvideo_kernel/_C)

