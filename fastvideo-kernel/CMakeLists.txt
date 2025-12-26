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
# If not defined, try to detect from PyTorch
if(NOT DEFINED TORCH_CUDA_ARCH_LIST)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import torch; print(';'.join([x.replace('sm_', '') for x in torch.cuda.get_arch_list()]))"
        OUTPUT_VARIABLE TORCH_CUDA_ARCH_LIST
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()

if(NOT TORCH_CUDA_ARCH_LIST)
    # Fallback default if detection failed or no GPU
    set(TORCH_CUDA_ARCH_LIST "9.0a")
endif()

message(STATUS "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}")

# Check if we are targeting Hopper (9.0a) for ThunderKittens
string(FIND "${TORCH_CUDA_ARCH_LIST}" "9.0a" HAS_HOPPER)
if(HAS_HOPPER EQUAL -1)
    message(STATUS "TORCH_CUDA_ARCH_LIST (${TORCH_CUDA_ARCH_LIST}) does not contain 9.0a. ThunderKittens kernels will be disabled.")
    set(ENABLE_TK_KERNELS OFF)
else()
    message(STATUS "Enabling ThunderKittens kernels (Hopper supported).")
    set(ENABLE_TK_KERNELS ON)
endif()

# Always try to build the extension if CUDA is available, but conditionally add sources/flags
set(BUILD_CXX_KERNELS ON)

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
    "-Xnvlink=--verbose"
    "-Xptxas=--verbose"
    "-Xptxas=--warn-on-spills"
)

# Only add -arch=sm_90a and TK flags if we are enabling TK kernels
if(ENABLE_TK_KERNELS)
    list(APPEND CUDA_FLAGS "-arch=sm_90a")
    list(APPEND CUDA_FLAGS "-DKITTENS_HOPPER")
endif()

if(BUILD_CXX_KERNELS)
    # Source files
    set(EXTENSION_SOURCES csrc/common_extension.cpp)
    
    # Conditionally add TK kernels
    if(ENABLE_TK_KERNELS)
        list(APPEND EXTENSION_SOURCES 
            csrc/attention/st_attn_h100.cu
            csrc/attention/block_sparse_h100.cu
        )
    endif()

    # Combined FastVideo Extension
    # Using name 'fastvideo_kernel_ops' to distinguish from the python package namespace
    Python_add_library(fastvideo_kernel_ops MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI
        ${EXTENSION_SOURCES}
    )
    
    # Build compile definitions list
    set(COMPILE_DEFS TORCH_EXTENSION_NAME=fastvideo_kernel_ops)
    if(ENABLE_TK_KERNELS)
        list(APPEND COMPILE_DEFS TK_COMPILE_ST_ATTN TK_COMPILE_BLOCK_SPARSE)
    endif()

    target_compile_definitions(fastvideo_kernel_ops PRIVATE ${COMPILE_DEFS})

    target_compile_options(fastvideo_kernel_ops PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>
    )
    
    # We install it to fastvideo_kernel/_C so we can load it to register the ops
    install(TARGETS fastvideo_kernel_ops LIBRARY DESTINATION fastvideo_kernel/_C)
endif()

