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

# ---------------------------
# ThunderKittens (TK) toggles
# ---------------------------
# AUTO: enable TK only when we can confidently target Hopper (sm_90a).
# ON:   force-enable TK kernels (intended for release wheels/images; does NOT require a GPU).
# OFF:  never build TK kernels.
set(FASTVIDEO_KERNEL_BUILD_TK "AUTO" CACHE STRING "Build ThunderKittens kernels: AUTO/ON/OFF")
set_property(CACHE FASTVIDEO_KERNEL_BUILD_TK PROPERTY STRINGS AUTO ON OFF)

# Prefer environment variable (used by CI) if CMake var is not explicitly set.
if(NOT DEFINED TORCH_CUDA_ARCH_LIST AND DEFINED ENV{TORCH_CUDA_ARCH_LIST})
    set(TORCH_CUDA_ARCH_LIST "$ENV{TORCH_CUDA_ARCH_LIST}")
endif()

message(STATUS "TORCH_CUDA_ARCH_LIST (cmake/env): ${TORCH_CUDA_ARCH_LIST}")
message(STATUS "FASTVIDEO_KERNEL_BUILD_TK: ${FASTVIDEO_KERNEL_BUILD_TK}")

set(ENABLE_TK_KERNELS OFF)
if(FASTVIDEO_KERNEL_BUILD_TK STREQUAL "ON")
    set(ENABLE_TK_KERNELS ON)
elseif(FASTVIDEO_KERNEL_BUILD_TK STREQUAL "OFF")
    set(ENABLE_TK_KERNELS OFF)
else()
    # AUTO: detect Hopper if possible.
    if(TORCH_CUDA_ARCH_LIST)
        # Accept common spellings: 9.0a, 90a, sm_90a.
        string(REGEX MATCH "(^|[; ,])((9\\.0a)|(90a)|(sm_90a))([; ,]|$)" _HAS_90A "${TORCH_CUDA_ARCH_LIST}")
        if(_HAS_90A)
            set(ENABLE_TK_KERNELS ON)
        endif()
    else()
        # Best-effort local detection (works when a CUDA device is visible).
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -c "import torch; import sys; \nprint('1' if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9) else '0')"
            OUTPUT_VARIABLE _LOCAL_HAS_HOPPER
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(_LOCAL_HAS_HOPPER STREQUAL "1")
            set(ENABLE_TK_KERNELS ON)
        endif()
    endif()
endif()

if(ENABLE_TK_KERNELS)
    message(STATUS "ThunderKittens kernels: ENABLED")
else()
    message(STATUS "ThunderKittens kernels: DISABLED (will use Triton fallbacks at runtime)")
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

# If TK is enabled, ensure we target Hopper. This is required even on GPU-less builders (CI).
if(ENABLE_TK_KERNELS)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
        set(CMAKE_CUDA_ARCHITECTURES "90a" CACHE STRING "CUDA architectures" FORCE)
    endif()
    list(APPEND CUDA_FLAGS "-DKITTENS_HOPPER")
    message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
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

