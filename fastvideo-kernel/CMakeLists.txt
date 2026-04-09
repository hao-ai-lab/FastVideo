cmake_minimum_required(VERSION 3.26 FATAL_ERROR)
project(fastvideo-kernel LANGUAGES CXX)

# Prefer environment variable (used by CI or pip install git+repo_addr) if CMake var is not explicitly set.
if(NOT DEFINED GPU_BACKEND AND DEFINED ENV{GPU_BACKEND})
    set(GPU_BACKEND "$ENV{GPU_BACKEND}")
endif()

if(GPU_BACKEND STREQUAL "ROCM")
    enable_language(HIP)
else()
    enable_language(CUDA)
    # Ensure CUDA toolkit targets (CUDA::cudart, CUDA::cuda_driver, etc.) are available.
    find_package(CUDAToolkit REQUIRED)
    if(NOT DEFINED CUDA_TOOLKIT_ROOT_DIR)
        if(DEFINED CUDAToolkit_ROOT)
            set(CUDA_TOOLKIT_ROOT_DIR "${CUDAToolkit_ROOT}" CACHE PATH
                "CUDA toolkit root directory" FORCE)
        elseif(DEFINED ENV{CUDAToolkit_ROOT})
            set(CUDA_TOOLKIT_ROOT_DIR "$ENV{CUDAToolkit_ROOT}" CACHE PATH
                "CUDA toolkit root directory" FORCE)
        elseif(DEFINED ENV{CUDA_HOME})
            set(CUDA_TOOLKIT_ROOT_DIR "$ENV{CUDA_HOME}" CACHE PATH
                "CUDA toolkit root directory" FORCE)
        endif()
    endif()
endif()

# Import common utils if needed, but we keep it simple for now

# Find Python and Torch
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Locate the installed torch package without importing it. This keeps CMake
# configure working even on nodes where CUDA runtime libraries are not yet on
# the dynamic loader path.
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_path('platlib'))"
    OUTPUT_VARIABLE PYTHON_PLATLIB
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_path('purelib'))"
    OUTPUT_VARIABLE PYTHON_PURELIB
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(TORCH_PYTHON_PACKAGE_DIR "")
foreach(_candidate
    "${PYTHON_PLATLIB}/torch"
    "${PYTHON_PURELIB}/torch"
)
    if(EXISTS "${_candidate}")
        set(TORCH_PYTHON_PACKAGE_DIR "${_candidate}")
        break()
    endif()
endforeach()

if(NOT TORCH_PYTHON_PACKAGE_DIR)
    message(FATAL_ERROR "Could not locate the installed torch Python package.")
endif()

list(APPEND TORCH_INCLUDE_DIRS
    "${TORCH_PYTHON_PACKAGE_DIR}/include"
    "${TORCH_PYTHON_PACKAGE_DIR}/include/torch/csrc/api/include"
)

if(NOT Torch_DIR)
    set(_TORCH_CONFIG_DIR "${TORCH_PYTHON_PACKAGE_DIR}/share/cmake/Torch")
    if(EXISTS "${_TORCH_CONFIG_DIR}/TorchConfig.cmake")
        set(Torch_DIR "${_TORCH_CONFIG_DIR}" CACHE PATH "Path to Torch CMake config" FORCE)
    endif()
endif()

# Find Torch package (still useful for libraries)
find_package(Torch REQUIRED)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/cutlass/include
    ${CMAKE_SOURCE_DIR}/include/tk/include
    ${CMAKE_SOURCE_DIR}/include/tk/prototype
    ${CMAKE_SOURCE_DIR}/csrc
    ${CMAKE_SOURCE_DIR}/csrc/turbodiffusion
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

set(_FASTVIDEO_KERNEL_BUILD_ATTN_QAT_INFER_DEFAULT "AUTO")
if(DEFINED FASTVIDEO_KERNEL_BUILD_MODIFIED_SAGE3 AND NOT DEFINED CACHE{FASTVIDEO_KERNEL_BUILD_ATTN_QAT_INFER})
    set(_FASTVIDEO_KERNEL_BUILD_ATTN_QAT_INFER_DEFAULT "${FASTVIDEO_KERNEL_BUILD_MODIFIED_SAGE3}")
endif()

set(FASTVIDEO_KERNEL_BUILD_ATTN_QAT_INFER "${_FASTVIDEO_KERNEL_BUILD_ATTN_QAT_INFER_DEFAULT}" CACHE STRING
    "Build attn_qat_infer Blackwell inference kernels: AUTO/ON/OFF")
set_property(CACHE FASTVIDEO_KERNEL_BUILD_ATTN_QAT_INFER PROPERTY STRINGS AUTO ON OFF)

if(DEFINED FASTVIDEO_KERNEL_BUILD_MODIFIED_SAGE3)
    message(DEPRECATION
        "FASTVIDEO_KERNEL_BUILD_MODIFIED_SAGE3 is deprecated. "
        "Use FASTVIDEO_KERNEL_BUILD_ATTN_QAT_INFER instead.")
endif()

# Prefer environment variable (used by CI) if CMake var is not explicitly set.
if(NOT DEFINED TORCH_CUDA_ARCH_LIST AND DEFINED ENV{TORCH_CUDA_ARCH_LIST})
    set(TORCH_CUDA_ARCH_LIST "$ENV{TORCH_CUDA_ARCH_LIST}")
endif()

message(STATUS "TORCH_CUDA_ARCH_LIST (cmake/env): ${TORCH_CUDA_ARCH_LIST}")
message(STATUS "FASTVIDEO_KERNEL_BUILD_TK: ${FASTVIDEO_KERNEL_BUILD_TK}")
message(STATUS "FASTVIDEO_KERNEL_BUILD_ATTN_QAT_INFER: ${FASTVIDEO_KERNEL_BUILD_ATTN_QAT_INFER}")

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
            COMMAND "${Python_EXECUTABLE}" -c "import torch; import sys; \nprint('1' if (torch.cuda.is_available() and torch.version.cuda and torch.cuda.get_device_capability()[0] >= 9) else '0')"
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

set(ENABLE_ATTN_QAT_INFER OFF)
if(GPU_BACKEND STREQUAL "ROCM")
    message(STATUS "attn_qat_infer kernels: DISABLED (ROCm build)")
else()
    set(_WANTS_ATTN_QAT_INFER OFF)
    if(FASTVIDEO_KERNEL_BUILD_ATTN_QAT_INFER STREQUAL "ON")
        set(_WANTS_ATTN_QAT_INFER ON)
    elseif(FASTVIDEO_KERNEL_BUILD_ATTN_QAT_INFER STREQUAL "AUTO")
        if(TORCH_CUDA_ARCH_LIST)
            string(REGEX MATCH
                "(^|[; ,])((12\\.0a)|(120a)|(sm_120a))([; ,]|$)"
                _HAS_120A "${TORCH_CUDA_ARCH_LIST}")
            if(_HAS_120A)
                set(_WANTS_ATTN_QAT_INFER ON)
            endif()
        else()
            execute_process(
                COMMAND "${Python_EXECUTABLE}" -c
                "import torch; print('1' if (torch.cuda.is_available() and torch.version.cuda and torch.cuda.get_device_capability()[0] >= 12) else '0')"
                OUTPUT_VARIABLE _LOCAL_HAS_BLACKWELL
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
            )
            if(_LOCAL_HAS_BLACKWELL STREQUAL "1")
                set(_WANTS_ATTN_QAT_INFER ON)
            endif()
        endif()
    endif()

    if(_WANTS_ATTN_QAT_INFER)
        if(CUDAToolkit_VERSION VERSION_LESS 12.8)
            message(WARNING
                "attn_qat_infer kernels require CUDA Toolkit 12.8+. "
                "Skipping because CUDAToolkit_VERSION=${CUDAToolkit_VERSION}.")
        else()
            set(ENABLE_ATTN_QAT_INFER ON)
        endif()
    endif()

    if(ENABLE_ATTN_QAT_INFER)
        message(STATUS "attn_qat_infer kernels: ENABLED")
    else()
        message(STATUS
            "attn_qat_infer kernels: DISABLED "
            "(requires CUDA 12.8+ and Blackwell sm_120a)")
    endif()
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
    set(EXTENSION_SOURCES 
        csrc/common_extension.cpp
        csrc/turbodiffusion/gemm/gemm.cu
        csrc/turbodiffusion/norm/rmsnorm.cu
        csrc/turbodiffusion/norm/layernorm.cu
        csrc/turbodiffusion/quant/quant.cu
    )
    
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

    # Link against Torch libraries to avoid undefined symbols at import time
    # (e.g., torch::autograd vtables) when loading the extension module.
    target_link_libraries(fastvideo_kernel_ops PRIVATE ${TORCH_LIBRARIES})

    # Also link against libtorch_python to satisfy Python-binding symbols
    # (e.g., torch::PyWarningHandler) required by torch/extension.h.
    file(GLOB TORCH_PYTHON_LIBRARY_CANDIDATES
        "${TORCH_PYTHON_PACKAGE_DIR}/lib/libtorch_python*"
    )
    list(LENGTH TORCH_PYTHON_LIBRARY_CANDIDATES _TORCH_PYTHON_LIBRARY_COUNT)
    if(_TORCH_PYTHON_LIBRARY_COUNT GREATER 0)
        list(GET TORCH_PYTHON_LIBRARY_CANDIDATES 0 TORCH_PYTHON_LIBRARY_PATH)
    else()
        set(TORCH_PYTHON_LIBRARY_PATH "")
    endif()
    if(TORCH_PYTHON_LIBRARY_PATH)
        message(STATUS "TORCH_PYTHON_LIBRARY_PATH: ${TORCH_PYTHON_LIBRARY_PATH}")
        target_link_libraries(fastvideo_kernel_ops PRIVATE "${TORCH_PYTHON_LIBRARY_PATH}")
    else()
        message(WARNING "Could not locate libtorch_python; fastvideo_kernel_ops may fail to import.")
    endif()

    # Link CUDA runtime + driver explicitly (fixes missing symbols like cuGetErrorString at import time)
    if(NOT GPU_BACKEND STREQUAL "ROCM")
        target_link_libraries(fastvideo_kernel_ops PRIVATE CUDA::cudart CUDA::cuda_driver)
    endif()

    # We install it to fastvideo_kernel/_C so we can load it to register the ops
    install(TARGETS fastvideo_kernel_ops LIBRARY DESTINATION fastvideo_kernel/_C)
endif()

if(ENABLE_ATTN_QAT_INFER)
    set(ATTN_QAT_INFER_DIR ${CMAKE_SOURCE_DIR}/attn_qat_infer)
    set(ATTN_QAT_INFER_INCLUDE_DIRS
        ${ATTN_QAT_INFER_DIR}
        ${CMAKE_SOURCE_DIR}/include/cutlass/include
        ${CMAKE_SOURCE_DIR}/include/cutlass/tools/util/include
        ${TORCH_INCLUDE_DIRS}
    )
    set(ATTN_QAT_INFER_CUDA_FLAGS
        "-O3"
        "-std=c++17"
        "-U__CUDA_NO_HALF_OPERATORS__"
        "-U__CUDA_NO_HALF_CONVERSIONS__"
        "-U__CUDA_NO_BFLOAT16_OPERATORS__"
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__"
        "-U__CUDA_NO_BFLOAT162_OPERATORS__"
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__"
        "--expt-relaxed-constexpr"
        "--expt-extended-lambda"
        "--use_fast_math"
        "--ptxas-options=--verbose,--warn-on-local-memory-usage"
        "-lineinfo"
        "-DCUTLASS_DEBUG_TRACE_LEVEL=0"
        "-DNDEBUG"
        "-DQBLKSIZE=128"
        "-DKBLKSIZE=128"
        "-DCTA256"
        "-DDQINRMEM"
    )

    Python_add_library(fp4attn_cuda MODULE WITH_SOABI
        attn_qat_infer/blackwell/api.cu
    )
    target_include_directories(fp4attn_cuda PRIVATE ${ATTN_QAT_INFER_INCLUDE_DIRS})
    target_compile_definitions(fp4attn_cuda PRIVATE TORCH_EXTENSION_NAME=fp4attn_cuda)
    target_compile_options(fp4attn_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -std=c++17>
        $<$<COMPILE_LANGUAGE:CUDA>:${ATTN_QAT_INFER_CUDA_FLAGS}>
    )
    set_target_properties(fp4attn_cuda PROPERTIES
        CUDA_ARCHITECTURES "120a"
        CXX_STANDARD 17
        CUDA_STANDARD 17
    )
    target_link_libraries(fp4attn_cuda PRIVATE ${TORCH_LIBRARIES} CUDA::cudart CUDA::cuda_driver)

    Python_add_library(fp4quant_cuda MODULE WITH_SOABI
        attn_qat_infer/quantization/fp4_quantization_4d.cu
    )
    target_include_directories(fp4quant_cuda PRIVATE ${ATTN_QAT_INFER_INCLUDE_DIRS})
    target_compile_definitions(fp4quant_cuda PRIVATE TORCH_EXTENSION_NAME=fp4quant_cuda)
    target_compile_options(fp4quant_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -std=c++17>
        $<$<COMPILE_LANGUAGE:CUDA>:${ATTN_QAT_INFER_CUDA_FLAGS}>
    )
    set_target_properties(fp4quant_cuda PROPERTIES
        CUDA_ARCHITECTURES "120a"
        CXX_STANDARD 17
        CUDA_STANDARD 17
    )
    target_link_libraries(fp4quant_cuda PRIVATE ${TORCH_LIBRARIES} CUDA::cudart CUDA::cuda_driver)

    if(TORCH_PYTHON_LIBRARY_PATH)
        target_link_libraries(fp4attn_cuda PRIVATE "${TORCH_PYTHON_LIBRARY_PATH}")
        target_link_libraries(fp4quant_cuda PRIVATE "${TORCH_PYTHON_LIBRARY_PATH}")
    endif()

    install(TARGETS fp4attn_cuda LIBRARY DESTINATION .)
    install(TARGETS fp4quant_cuda LIBRARY DESTINATION .)
endif()
