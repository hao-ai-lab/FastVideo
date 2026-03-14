from fastvideo.attention.yunchang.ring import (
    ring_flash_attn_func, ring_flash_attn_qkvpacked_func,
    zigzag_ring_flash_attn_func, zigzag_ring_flash_attn_qkvpacked_func,
    stripe_flash_attn_func, stripe_flash_attn_qkvpacked_func,
    ring_pytorch_attn_func, ring_flashinfer_attn_func,
    ring_flashinfer_attn_qkvpacked_func, ring_npu_flash_attn_func)

RING_IMPL_DICT = {
    "basic": ring_flash_attn_func,
    "zigzag": zigzag_ring_flash_attn_func,
    "strip": stripe_flash_attn_func,
    "basic_pytorch": ring_pytorch_attn_func,
    "basic_flashinfer": ring_flashinfer_attn_func,
    "basic_npu": ring_npu_flash_attn_func
}

RING_IMPL_QKVPACKED_DICT = {
    "basic": ring_flash_attn_qkvpacked_func,
    "zigzag": zigzag_ring_flash_attn_qkvpacked_func,
    "strip": stripe_flash_attn_qkvpacked_func,
    "basic_flashinfer": ring_flashinfer_attn_qkvpacked_func,
}
