try: 
    from flash_attn_interface import flash_attn_func as flash_attn_func_v3 
    has_v3 = True
except: 
    pass
from flash_attn import flash_attn_varlen_qkvpacked_func
from flash_attn.bert_padding import pad_input, unpad_input
from einops import rearrange


def flash_attn_no_pad(
    qkv, key_padding_mask, causal=False, dropout_p=0.0, softmax_scale=None, attn_impl="fa3",
):
    # adapted from https://github.com/Dao-AILab/flash-attention/blob/13403e81157ba37ca525890f2f0f2137edf75311/flash_attn/flash_attention.py#L27
    batch_size = qkv.shape[0]
    seqlen = qkv.shape[1]
    nheads = qkv.shape[-2]
    x = rearrange(qkv, "b s three h d -> b s (three h d)")
    x_unpad, indices, cu_seqlens, max_s, used_seqlens_in_batch = unpad_input(
        x, key_padding_mask
    )

    x_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads)
    if attn_impl == "fa3":
        assert has_v3
        q, k, v = x_unpad[:, 0].unsqueeze(0), x_unpad[:, 1].unsqueeze(0), x_unpad[:, 2].unsqueeze(0)
        assert dropout_p == 0.0
        output_unpad = flash_attn_func_v3(
            q,
            k,
            v,
        )[0].squeeze(0)
    else:
        output_unpad = flash_attn_varlen_qkvpacked_func(
            x_unpad,
            cu_seqlens,
            max_s,
            dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
        )
    output = rearrange(
        pad_input(
            rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, batch_size, seqlen
        ),
        "b s (h d) -> b s h d",
        h=nheads,
    )
    return output
