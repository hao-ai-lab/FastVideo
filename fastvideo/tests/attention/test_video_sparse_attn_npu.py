# SPDX-License-Identifier: Apache-2.0
"""Dispatch and import contracts; numeric cases live in benchmarks/vsa_npu."""
import subprocess
import sys
import os

import pytest
import torch

from fastvideo.platforms.interface import AttentionBackendEnum
from fastvideo.platforms.npu import NPUPlatform
from fastvideo.attention.backends.video_sparse_attn_npu import NPUVideoSparseAttentionImpl, tiled_forward


def test_npu_vsa_explicit_dispatch_and_unsupported_domains():
    selected=AttentionBackendEnum.VIDEO_SPARSE_ATTN
    assert NPUPlatform.get_attn_backend_cls(selected,128,torch.bfloat16).endswith('NPUVideoSparseAttentionBackend')
    for dtype,dim in [(torch.float16,128),(torch.float32,128),(torch.bfloat16,80)]:
        with pytest.raises(NotImplementedError):
            NPUPlatform.get_attn_backend_cls(selected,dim,dtype)
    assert NPUPlatform.get_attn_backend_cls(None,128,torch.bfloat16).endswith('SDPABackend')


def test_npu_vsa_module_does_not_load_kernel_packages():
    code='''
import sys
import fastvideo.attention.backends.video_sparse_attn_npu
assert 'fastvideo_kernel' not in sys.modules
assert 'fastvideo.attention.backends.video_sparse_attn_npu_kernels' not in sys.modules
'''
    subprocess.run([sys.executable,'-c',code],check=True)


def test_npu_vsa_constructor_rejects_unsupported_metadata(monkeypatch):
    from types import SimpleNamespace
    import fastvideo.distributed.parallel_state as parallel
    import fastvideo.attention.backends.video_sparse_attn as upstream
    monkeypatch.setattr(parallel,'model_parallel_is_initialized',lambda:True)
    monkeypatch.setattr(parallel,'get_sp_world_size',lambda:4)
    monkeypatch.setattr(upstream,'get_sp_group',lambda:SimpleNamespace(world_size=4))
    NPUVideoSparseAttentionImpl(12,64,False,64**-.5)
    for heads in [0,10]:
        with pytest.raises(NotImplementedError,match='divisible'):
            NPUVideoSparseAttentionImpl(heads,64,False,64**-.5)
    monkeypatch.setattr(parallel,'get_sp_world_size',lambda:2)
    with pytest.raises(NotImplementedError,match='SP1/SP4'):
        NPUVideoSparseAttentionImpl(12,64,False,64**-.5)
    for causal,scale,kv in [(True,64**-.5,None),(False,.1,None),(False,64**-.5,2)]:
        with pytest.raises(NotImplementedError):
            NPUVideoSparseAttentionImpl(12,64,causal,scale,num_kv_heads=kv)


def test_npu_vsa_import_without_platform_extensions():
    code='''
import importlib.abc
import importlib.util
import sys
real_find_spec=importlib.util.find_spec
def find_spec(name,*args,**kwargs):
    if name.split('.')[0] in {'torch_npu','fastvideo_kernel'}:return None
    return real_find_spec(name,*args,**kwargs)
importlib.util.find_spec=find_spec
class MissingPlatformExtensions(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch_npu','fastvideo_kernel'}:
            raise ModuleNotFoundError('platform extension unavailable in CPU import test')
sys.meta_path.insert(0,MissingPlatformExtensions())
import fastvideo.attention.backends.video_sparse_attn_npu
assert 'torch_npu' not in sys.modules
assert 'fastvideo_kernel' not in sys.modules
assert 'fastvideo.attention.backends.video_sparse_attn_npu_kernels' not in sys.modules
'''
    env=os.environ.copy();env['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
    subprocess.run([sys.executable,'-c',code],check=True,env=env)


def test_npu_vsa_rejects_cpu_instead_of_dense_fallback():
    x=torch.zeros(1,1,64,64,dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError,match='NPU BF16'):
        tiled_forward(x,x,x,None,torch.tensor([64]),.8)


def test_npu_vsa_rejects_backward():
    if not hasattr(torch,'npu') or not torch.npu.is_available():
        pytest.skip('requires Ascend NPU')
    x=torch.zeros(1,1,64,64,device='npu',dtype=torch.bfloat16,requires_grad=True)
    with pytest.raises(NotImplementedError,match='backward'):
        tiled_forward(x,x,x,None,torch.tensor([64],device='npu'),.8)
