cmake_minimum_required(VERSION 3.26 FATAL_ERROR)
project(fastvideo-kernel LANGUAGES CXX)

# Capture any caller-provided -DCMAKE_CUDA_ARCHITECTURES *before* enable_language(CUDA)
# auto-populates it with CMake's built-in default (an old arch, e.g. sm_75 on CUDA 13).
# torch's cmake actually ignores CMAKE_CUDA_ARCHITECTURES (it drives arch selection via
# TORCH_CUDA_ARCH_LIST), so we only use this captured value to honor an explicit pin by
# translating it into TORCH_CUDA_ARCH_LIST below.
set(_FASTVIDEO_USER_CUDA_ARCH "${CMAKE_CUDA_ARCHITECTURES}")

# Prefer environment variable (used by CI or uv pip install git+repo_addr) if CMake var is not explicitly set.
if(NOT DEFINED GPU_BACKEND AND DEFINED ENV{GPU_BACKEND})
    set(GPU_BACKEND "$ENV{GPU_BACKEND}")
endif()

if(GPU_BACKEND STREQUAL "ROCM")
    enable_language(HIP)
else()
    enable_language(CUDA)
    # Ensure CUDA toolkit targets (CUDA::cudart, CUDA::cuda_driver, etc.) are available.
    find_package(CUDAToolkit REQUIRED)
endif()

# Import common utils if needed, but we keep it simple for now

# Find Python and Torch
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# ---------------------------------------------------------------------------
# Resolve the target CUDA architecture, BEFORE find_package(Torch) below.
#
# torch's cmake (Caffe2 public/cuda.cmake) takes over arch selection: it emits
# the real -gencode flags from TORCH_CUDA_ARCH_LIST and forces
# CMAKE_CUDA_ARCHITECTURES to OFF. So the *effective* arch is whatever
# TORCH_CUDA_ARCH_LIST is when find_package(Torch) runs. build.sh exports it;
# standards-based builds (pip / uv pip install, sdist) don't, and torch then
# auto-detects an arch that does not match the GPU -- the kernels build but fail
# at runtime ("no kernel image is available for execution on the device").
# Resolve it here when absent (mirrors build.sh): honor a pinned
# CMAKE_CUDA_ARCHITECTURES if given, else probe the visible GPU with torch.
# ---------------------------------------------------------------------------
if(NOT GPU_BACKEND STREQUAL "ROCM")
    if(DEFINED ENV{TORCH_CUDA_ARCH_LIST})
        message(STATUS "CUDA arch: TORCH_CUDA_ARCH_LIST=$ENV{TORCH_CUDA_ARCH_LIST} (from environment)")
    elseif(TORCH_CUDA_ARCH_LIST)
        set(ENV{TORCH_CUDA_ARCH_LIST} "${TORCH_CUDA_ARCH_LIST}")
        message(STATUS "CUDA arch: TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} (from cmake)")
    else()
        set(_FV_ARCH_LIST "")
        if(_FASTVIDEO_USER_CUDA_ARCH)
            # Caller pinned -DCMAKE_CUDA_ARCHITECTURES (which torch ignores); translate it
            # to the TORCH_CUDA_ARCH_LIST spelling: "121" -> "12.1", "90a" -> "9.0a".
            foreach(_fv_arch IN LISTS _FASTVIDEO_USER_CUDA_ARCH)
                string(REGEX MATCH "[af]$" _fv_suffix "${_fv_arch}")
                string(REGEX REPLACE "[af]$" "" _fv_num "${_fv_arch}")
                string(REGEX REPLACE "(.)$" ".\\1" _fv_num "${_fv_num}")    # dot before the last digit
                list(APPEND _FV_ARCH_LIST "${_fv_num}${_fv_suffix}")
            endforeach()
            message(STATUS "CUDA arch: TORCH_CUDA_ARCH_LIST=${_FV_ARCH_LIST} (from -DCMAKE_CUDA_ARCHITECTURES=${_FASTVIDEO_USER_CUDA_ARCH})")
        else()
            # Best-effort probe of the visible GPU (mirrors build.sh detect_with_torch).
            execute_process(
                COMMAND "${Python_EXECUTABLE}" -c "import torch; assert torch.cuda.is_available(); mj, mn = torch.cuda.get_device_capability(0); print(f'{mj}.{mn}a' if (mj, mn) == (9, 0) else f'{mj}.{mn}')"
                OUTPUT_VARIABLE _FV_ARCH_LIST
                OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE _fv_detect_rc
                ERROR_QUIET
            )
            if(_fv_detect_rc EQUAL 0 AND _FV_ARCH_LIST)
                message(STATUS "CUDA arch: TORCH_CUDA_ARCH_LIST=${_FV_ARCH_LIST} (detected via torch, live GPU)")
            else()
                set(_FV_ARCH_LIST "")
            endif()
        endif()

        if(_FV_ARCH_LIST)
            set(TORCH_CUDA_ARCH_LIST "${_FV_ARCH_LIST}")
            set(ENV{TORCH_CUDA_ARCH_LIST} "${_FV_ARCH_LIST}")
        else()
            message(FATAL_ERROR
                "fastvideo-kernel: could not determine the target CUDA architecture.\n"
                "Refusing to let torch auto-detect an arch that may not run on this GPU. "
                "Fix with one of:\n"
                "  - set TORCH_CUDA_ARCH_LIST (e.g. 12.1, or 9.0a for Hopper), or\n"
                "  - pass -DCMAKE_CUDA_ARCHITECTURES=<arch> (e.g. 121), or\n"
                "  - build where the target GPU is visible to torch.\n"
                "Note: 'pip/uv pip install' builds under build isolation, which hides the "
                "GPU; set TORCH_CUDA_ARCH_LIST or add --no-build-isolation. "
                "fastvideo-kernel/build.sh sets all of this for you.")
        endif()
    endif()
endif()

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

# The pinned ThunderKittens revision predates its ARM compatibility fix. On
# AArch64, CUDA's device frontend treats plain `char` as unsigned, so the two
# `char2{i, i}` initializers in base_types.cuh are rejected as narrowing to the
# signed-char fields of CUDA's char2. Backport the two explicit casts proposed in
# https://github.com/HazyResearch/ThunderKittens/pull/179 into a build-tree copy
# so the pinned submodule remains pristine. Keeping this in CMake also covers
# direct pip/uv builds that do not go through build.sh.
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _FASTVIDEO_SYSTEM_PROCESSOR)
if(ENABLE_TK_KERNELS AND _FASTVIDEO_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    set(_TK_ARM_COMPAT_DIR "${CMAKE_CURRENT_BINARY_DIR}/tk-arm-compat")
    set(_TK_ARM_INCLUDE_DIR "${_TK_ARM_COMPAT_DIR}/include")
    file(REMOVE_RECURSE "${_TK_ARM_COMPAT_DIR}")
    file(COPY "${CMAKE_SOURCE_DIR}/include/tk/include" DESTINATION "${_TK_ARM_COMPAT_DIR}")

    set(_TK_ARM_BASE_TYPES "${_TK_ARM_INCLUDE_DIR}/common/base_types.cuh")
    file(READ "${_TK_ARM_BASE_TYPES}" _TK_ARM_BASE_TYPES_CONTENT)
    string(REGEX MATCHALL "return char2[{]i, i[}]" _TK_ARM_CHAR_PACKS "${_TK_ARM_BASE_TYPES_CONTENT}")
    list(LENGTH _TK_ARM_CHAR_PACKS _TK_ARM_CHAR_PACK_COUNT)
    if(NOT _TK_ARM_CHAR_PACK_COUNT EQUAL 2)
        message(FATAL_ERROR
            "ThunderKittens ARM compatibility patch expected two char2{i, i} initializers; "
            "found ${_TK_ARM_CHAR_PACK_COUNT}. Update or remove the backport for the pinned revision.")
    endif()
    string(REPLACE
        "return char2{i, i}"
        "return char2{static_cast<signed char>(i), static_cast<signed char>(i)}"
        _TK_ARM_BASE_TYPES_CONTENT
        "${_TK_ARM_BASE_TYPES_CONTENT}")
    file(WRITE "${_TK_ARM_BASE_TYPES}" "${_TK_ARM_BASE_TYPES_CONTENT}")
    message(STATUS "Applied ThunderKittens ARM signed-char compatibility backport")
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

# ---------------------------------------------------------------------------
# Per-arch split for the Blackwell FP4 (attn_qat_infer) build
# ---------------------------------------------------------------------------
# The FP4 kernels are sm_120a-only (they emit `cvt.e2m1x2` etc.), while the main
# extension (Hopper-only TK + generic turbodiffusion) targets the full arch list.
# find_package(Torch) injects ONE global -gencode list into CMAKE_CUDA_FLAGS that
# forces every target onto every arch, so the FP4 sources also get the sm_90a pass
# and ptxas rejects their Blackwell instructions. Strip that global list and drive
# arch per target via CUDA_ARCHITECTURES instead (the fp4* targets pin 120a below;
# the main extension gets the full list). Only do this for the FP4 build with an
# explicit arch list, so the cu126 / local autodetect paths stay untouched.
if(ENABLE_ATTN_QAT_INFER AND TORCH_CUDA_ARCH_LIST)
    message(STATUS "[per-arch] CMAKE_CUDA_FLAGS before strip: ${CMAKE_CUDA_FLAGS}")
    string(REGEX REPLACE "-gencode[ =]+arch=[^ ]+" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
    string(REGEX REPLACE " +" " " CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
    message(STATUS "[per-arch] CMAKE_CUDA_FLAGS after strip:  ${CMAKE_CUDA_FLAGS}")
    # Convert TORCH_CUDA_ARCH_LIST ("9.0a;12.0a") to CMake form ("90a;120a").
    set(FASTVIDEO_MAIN_CUDA_ARCHS "${TORCH_CUDA_ARCH_LIST}")
    string(REPLACE "sm_" "" FASTVIDEO_MAIN_CUDA_ARCHS "${FASTVIDEO_MAIN_CUDA_ARCHS}")
    string(REPLACE "." "" FASTVIDEO_MAIN_CUDA_ARCHS "${FASTVIDEO_MAIN_CUDA_ARCHS}")
    message(STATUS "[per-arch] main extension archs=${FASTVIDEO_MAIN_CUDA_ARCHS}, fp4* archs=120a")
endif()

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
    # ARM/aarch64 defaults `char` to unsigned, but ThunderKittens headers assume the
    # x86 signed-char behavior (else base_types.cuh hits "narrowing conversion from
    # char to signed char"). Force signed char so TK compiles on Grace Hopper; this
    # is a no-op on x86_64, where char is already signed.
    "-Xcompiler=-fsigned-char"
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

    if(_TK_ARM_INCLUDE_DIR)
        target_include_directories(fastvideo_kernel_ops BEFORE PRIVATE "${_TK_ARM_INCLUDE_DIR}")
    endif()

    # When the per-arch split is active (FP4 build), torch's global gencode was
    # stripped above, so set this target's arch explicitly. TK is guarded down to
    # sm_90a in source; the turbodiffusion kernels are generic, so the main
    # extension targets the full list. (No-FP4 builds keep torch's global gencode.)
    if(ENABLE_ATTN_QAT_INFER AND FASTVIDEO_MAIN_CUDA_ARCHS)
        set_target_properties(fastvideo_kernel_ops PROPERTIES
            CUDA_ARCHITECTURES "${FASTVIDEO_MAIN_CUDA_ARCHS}")
    endif()

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
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import torch; from pathlib import Path; p=Path(torch.__file__).parent/'lib'; m=sorted(p.glob('libtorch_python*')); print(str(m[0]) if m else '')"
        OUTPUT_VARIABLE TORCH_PYTHON_LIBRARY_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
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
