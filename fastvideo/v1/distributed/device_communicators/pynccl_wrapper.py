# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/distributed/device_communicators/pymccl_wrapper.py

# This file is a pure Python wrapper for the MCCL library.
# The main purpose is to use MCCL combined with MUSA graph.
# Before writing this script, we tried the following approach:
# 1. We tried to use `cupy`, it calls MCCL correctly, but `cupy` itself
#  often gets stuck when initializing the MCCL communicator.
# 2. We tried to use `torch.distributed`, but `torch.distributed.all_reduce`
#  contains many other potential musa APIs, that are not allowed during
#  capturing the MUSA graph. For further details, please check
# https://discuss.pytorch.org/t/pytorch-musagraph-with-mccl-operation-failed/ .
#
# Another rejected idea is to write a C/C++ binding for MCCL. It is usually
# doable, but we often encounter issues related with mccl versions, and need
# to switch between different versions of MCCL. See
# https://github.com/NVIDIA/mccl/issues/1234 for more details.
# A C/C++ binding is not flexible enough to handle this. It requires
# recompilation of the code every time we want to switch between different
# versions. This current implementation, with a **pure** Python wrapper, is
# more flexible. We can easily switch between different versions of MCCL by
# changing the environment variable `FASTVIDEO_MCCL_SO_PATH`, or the `so_file`
# variable in the code.

#TODO(will): support FASTVIDEO_MCCL_SO_PATH

import ctypes
import platform
from dataclasses import dataclass
from typing import Any

import torch
from torch.distributed import ReduceOp

from fastvideo.v1.logger import init_logger
from fastvideo.v1.utils import find_mccl_library

logger = init_logger(__name__)

# === export types and functions from mccl to Python ===
# for the original mccl definition, please check
# https://github.com/NVIDIA/mccl/blob/master/src/mccl.h.in

mcclResult_t = ctypes.c_int
mcclComm_t = ctypes.c_void_p


class mcclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


musaStream_t = ctypes.c_void_p
buffer_type = ctypes.c_void_p

mcclDataType_t = ctypes.c_int


class mcclDataTypeEnum:
    mcclInt8 = 0
    mcclChar = 0
    mcclUint8 = 1
    mcclInt32 = 2
    mcclInt = 2
    mcclUint32 = 3
    mcclInt64 = 4
    mcclUint64 = 5
    mcclFloat16 = 6
    mcclHalf = 6
    mcclFloat32 = 7
    mcclFloat = 7
    mcclFloat64 = 8
    mcclDouble = 8
    mcclBfloat16 = 9
    mcclNumTypes = 10

    @classmethod
    def from_torch(cls, dtype: torch.dtype) -> int:
        if dtype == torch.int8:
            return cls.mcclInt8
        if dtype == torch.uint8:
            return cls.mcclUint8
        if dtype == torch.int32:
            return cls.mcclInt32
        if dtype == torch.int64:
            return cls.mcclInt64
        if dtype == torch.float16:
            return cls.mcclFloat16
        if dtype == torch.float32:
            return cls.mcclFloat32
        if dtype == torch.float64:
            return cls.mcclFloat64
        if dtype == torch.bfloat16:
            return cls.mcclBfloat16
        raise ValueError(f"Unsupported dtype: {dtype}")


mcclRedOp_t = ctypes.c_int


class mcclRedOpTypeEnum:
    mcclSum = 0
    mcclProd = 1
    mcclMax = 2
    mcclMin = 3
    mcclAvg = 4
    mcclNumOps = 5

    @classmethod
    def from_torch(cls, op: ReduceOp) -> int:
        if op == ReduceOp.SUM:
            return cls.mcclSum
        if op == ReduceOp.PRODUCT:
            return cls.mcclProd
        if op == ReduceOp.MAX:
            return cls.mcclMax
        if op == ReduceOp.MIN:
            return cls.mcclMin
        if op == ReduceOp.AVG:
            return cls.mcclAvg
        raise ValueError(f"Unsupported op: {op}")


@dataclass
class Function:
    name: str
    restype: Any
    argtypes: list[Any]


class MCCLLibrary:
    exported_functions = [
        # const char* mcclGetErrorString(mcclResult_t result)
        Function("mcclGetErrorString", ctypes.c_char_p, [mcclResult_t]),
        # mcclResult_t  mcclGetVersion(int *version);
        Function("mcclGetVersion", mcclResult_t,
                 [ctypes.POINTER(ctypes.c_int)]),
        # mcclResult_t mcclGetUniqueId(mcclUniqueId* uniqueId);
        Function("mcclGetUniqueId", mcclResult_t,
                 [ctypes.POINTER(mcclUniqueId)]),
        # mcclResult_t  mcclCommInitRank(
        #   mcclComm_t* comm, int nranks, mcclUniqueId commId, int rank);
        # note that mcclComm_t is a pointer type, so the first argument
        # is a pointer to a pointer
        Function("mcclCommInitRank", mcclResult_t, [
            ctypes.POINTER(mcclComm_t), ctypes.c_int, mcclUniqueId, ctypes.c_int
        ]),
        # mcclResult_t  mcclAllReduce(
        #   const void* sendbuff, void* recvbuff, size_t count,
        #   mcclDataType_t datatype, mcclRedOp_t op, mcclComm_t comm,
        #   musaStream_t stream);
        # note that musaStream_t is a pointer type, so the last argument
        # is a pointer
        Function("mcclAllReduce", mcclResult_t, [
            buffer_type, buffer_type, ctypes.c_size_t, mcclDataType_t,
            mcclRedOp_t, mcclComm_t, musaStream_t
        ]),

        # mcclResult_t  mcclAllGather(
        #   const void* sendbuff, void* recvbuff, size_t count,
        #   mcclDataType_t datatype, mcclComm_t comm,
        #   musaStream_t stream);
        # note that musaStream_t is a pointer type, so the last argument
        # is a pointer
        Function("mcclAllGather", mcclResult_t, [
            buffer_type, buffer_type, ctypes.c_size_t, mcclDataType_t,
            mcclComm_t, musaStream_t
        ]),

        # mcclResult_t  mcclReduceScatter(
        #   const void* sendbuff, void* recvbuff, size_t count,
        #   mcclDataType_t datatype, mcclRedOp_t op, mcclComm_t comm,
        #   musaStream_t stream);
        # note that musaStream_t is a pointer type, so the last argument
        # is a pointer
        Function("mcclReduceScatter", mcclResult_t, [
            buffer_type, buffer_type, ctypes.c_size_t, mcclDataType_t,
            mcclRedOp_t, mcclComm_t, musaStream_t
        ]),

        # mcclResult_t  mcclSend(
        #   const void* sendbuff, size_t count, mcclDataType_t datatype,
        #   int dest, mcclComm_t comm, musaStream_t stream);
        Function("mcclSend", mcclResult_t, [
            buffer_type, ctypes.c_size_t, mcclDataType_t, ctypes.c_int,
            mcclComm_t, musaStream_t
        ]),

        # mcclResult_t  mcclRecv(
        #   void* recvbuff, size_t count, mcclDataType_t datatype,
        #   int src, mcclComm_t comm, musaStream_t stream);
        Function("mcclRecv", mcclResult_t, [
            buffer_type, ctypes.c_size_t, mcclDataType_t, ctypes.c_int,
            mcclComm_t, musaStream_t
        ]),

        # mcclResult_t mcclBroadcast(
        #   const void* sendbuff, void* recvbuff, size_t count,
        #   mcclDataType_t datatype, int root, mcclComm_t comm,
        #   musaStream_t stream);
        Function("mcclBroadcast", mcclResult_t, [
            buffer_type, buffer_type, ctypes.c_size_t, mcclDataType_t,
            ctypes.c_int, mcclComm_t, musaStream_t
        ]),

        # be cautious! this is a collective call, it will block until all
        # processes in the communicator have called this function.
        # because Python object destruction can happen in random order,
        # it is better not to call it at all.
        # mcclResult_t  mcclCommDestroy(mcclComm_t comm);
        Function("mcclCommDestroy", mcclResult_t, [mcclComm_t]),
    ]

    # class attribute to store the mapping from the path to the library
    # to avoid loading the same library multiple times
    path_to_library_cache: dict[str, Any] = {}

    # class attribute to store the mapping from library path
    #  to the corresponding dictionary
    path_to_dict_mapping: dict[str, dict[str, Any]] = {}

    def __init__(self, so_file: str | None = None):

        so_file = so_file or find_mccl_library()

        try:
            if so_file not in MCCLLibrary.path_to_dict_mapping:
                lib = ctypes.CDLL(so_file)
                MCCLLibrary.path_to_library_cache[so_file] = lib
            self.lib = MCCLLibrary.path_to_library_cache[so_file]
        except Exception as e:
            logger.error(
                "Failed to load MCCL library from %s ."
                "It is expected if you are not running on NVIDIA/AMD GPUs."
                "Otherwise, the mccl library might not exist, be corrupted "
                "or it does not support the current platform %s."
                "If you already have the library, please set the "
                "environment variable FASTVIDEO_MCCL_SO_PATH"
                " to point to the correct mccl library path.", so_file,
                platform.platform())
            raise e

        if so_file not in MCCLLibrary.path_to_dict_mapping:
            _funcs: dict[str, Any] = {}
            for func in MCCLLibrary.exported_functions:
                f = getattr(self.lib, func.name)
                f.restype = func.restype
                f.argtypes = func.argtypes
                _funcs[func.name] = f
            MCCLLibrary.path_to_dict_mapping[so_file] = _funcs
        self._funcs = MCCLLibrary.path_to_dict_mapping[so_file]

    def mcclGetErrorString(self, result: mcclResult_t) -> str:
        return str(self._funcs["mcclGetErrorString"](result).decode("utf-8"))

    def MCCL_CHECK(self, result: mcclResult_t) -> None:
        if result != 0:
            error_str = self.mcclGetErrorString(result)
            raise RuntimeError(f"MCCL error: {error_str}")

    def mcclGetVersion(self) -> str:
        version = ctypes.c_int()
        self.MCCL_CHECK(self._funcs["mcclGetVersion"](ctypes.byref(version)))
        version_str = str(version.value)
        # something like 21903 --> "2.19.3"
        major = version_str[0].lstrip("0")
        minor = version_str[1:3].lstrip("0")
        patch = version_str[3:].lstrip("0")
        return f"{major}.{minor}.{patch}"

    def mcclGetUniqueId(self) -> mcclUniqueId:
        unique_id = mcclUniqueId()
        self.MCCL_CHECK(self._funcs["mcclGetUniqueId"](ctypes.byref(unique_id)))
        return unique_id

    def mcclCommInitRank(self, world_size: int, unique_id: mcclUniqueId,
                         rank: int) -> mcclComm_t:
        comm = mcclComm_t()
        self.MCCL_CHECK(self._funcs["mcclCommInitRank"](ctypes.byref(comm),
                                                        world_size, unique_id,
                                                        rank))
        return comm

    def mcclAllReduce(self, sendbuff: buffer_type, recvbuff: buffer_type,
                      count: int, datatype: int, op: int, comm: mcclComm_t,
                      stream: musaStream_t) -> None:
        # `datatype` actually should be `mcclDataType_t`
        # and `op` should be `mcclRedOp_t`
        # both are aliases of `ctypes.c_int`
        # when we pass int to a function, it will be converted to `ctypes.c_int`
        # by ctypes automatically
        self.MCCL_CHECK(self._funcs["mcclAllReduce"](sendbuff, recvbuff, count,
                                                     datatype, op, comm,
                                                     stream))

    def mcclReduceScatter(self, sendbuff: buffer_type, recvbuff: buffer_type,
                          count: int, datatype: int, op: int, comm: mcclComm_t,
                          stream: musaStream_t) -> None:
        # `datatype` actually should be `mcclDataType_t`
        # and `op` should be `mcclRedOp_t`
        # both are aliases of `ctypes.c_int`
        # when we pass int to a function, it will be converted to `ctypes.c_int`
        # by ctypes automatically
        self.MCCL_CHECK(self._funcs["mcclReduceScatter"](sendbuff, recvbuff,
                                                         count, datatype, op,
                                                         comm, stream))

    def mcclAllGather(self, sendbuff: buffer_type, recvbuff: buffer_type,
                      count: int, datatype: int, comm: mcclComm_t,
                      stream: musaStream_t) -> None:
        # `datatype` actually should be `mcclDataType_t`
        # which is an aliases of `ctypes.c_int`
        # when we pass int to a function, it will be converted to `ctypes.c_int`
        # by ctypes automatically
        self.MCCL_CHECK(self._funcs["mcclAllGather"](sendbuff, recvbuff, count,
                                                     datatype, comm, stream))

    def mcclSend(self, sendbuff: buffer_type, count: int, datatype: int,
                 dest: int, comm: mcclComm_t, stream: musaStream_t) -> None:
        self.MCCL_CHECK(self._funcs["mcclSend"](sendbuff, count, datatype, dest,
                                                comm, stream))

    def mcclRecv(self, recvbuff: buffer_type, count: int, datatype: int,
                 src: int, comm: mcclComm_t, stream: musaStream_t) -> None:
        self.MCCL_CHECK(self._funcs["mcclRecv"](recvbuff, count, datatype, src,
                                                comm, stream))

    def mcclBroadcast(self, sendbuff: buffer_type, recvbuff: buffer_type,
                      count: int, datatype: int, root: int, comm: mcclComm_t,
                      stream: musaStream_t) -> None:
        self.MCCL_CHECK(self._funcs["mcclBroadcast"](sendbuff, recvbuff, count,
                                                     datatype, root, comm,
                                                     stream))

    def mcclCommDestroy(self, comm: mcclComm_t) -> None:
        self.MCCL_CHECK(self._funcs["mcclCommDestroy"](comm))


__all__ = [
    "MCCLLibrary", "mcclDataTypeEnum", "mcclRedOpTypeEnum", "mcclUniqueId",
    "mcclComm_t", "musaStream_t", "buffer_type"
]
