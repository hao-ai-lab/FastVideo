from einops import rearrange
from flash_attn import flash_attn_varlen_qkvpacked_func,flash_attn_qkvpacked_func
from flash_attn.bert_padding import pad_input, unpad_input


def flash_attn_no_pad(qkv, key_padding_mask, causal=False, dropout_p=0.0, softmax_scale=None):
    # adapted from https://github.com/Dao-AILab/flash-attention/blob/13403e81157ba37ca525890f2f0f2137edf75311/flash_attn/flash_attention.py#L27
    batch_size = qkv.shape[0]
    seqlen = qkv.shape[1]
    nheads = qkv.shape[-2]
    #print(f"===== b, s, nheads: {batch_size}, {seqlen}, {nheads} =====")
    #x = rearrange(qkv, "b s three h d -> b s (three h d)")
    #print(f"===== qkv shape: {qkv.shape}=====")
    #print(f"===== x shape: {x.shape} =====")
    # x_unpad, indices, cu_seqlens, max_s, used_seqlens_in_batch = unpad_input(x, key_padding_mask)
    #x_unpad, indices, cu_seqlens, max_s = unpad_input(x, key_padding_mask)
    #x_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads)
    #print(f"===== x_unpad shape: {x_unpad.shape}")
    #output_unpad = flash_attn_varlen_qkvpacked_func(
    #    x_unpad,
    #    cu_seqlens,
    #    max_s,
    #    dropout_p,
    #    softmax_scale=softmax_scale,
    #    causal=causal,
    #)
    output_unpad = flash_attn_qkvpacked_func(
            qkv,
            dropout_p,
            softmax_scale,
            causal
    )

    #print(f"===== output_unpad shape: {output_unpad.shape} =====")
    #output = rearrange(
    #    pad_input(rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, batch_size, seqlen),
    #    "b s (h d) -> b s h d",
    #    h=nheads,
    #)
    return output_unpad
