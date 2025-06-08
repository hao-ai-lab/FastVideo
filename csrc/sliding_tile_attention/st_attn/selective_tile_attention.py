import torch

BLOCK_M = 196
BLOCK_N = 128


def get_block_sparse_mask(q, k, simthreshd1=0.1, cdfthreshd=0.9, return_sparsity=False):
    assert q.shape == k.shape
    b, h, l, d = q.shape
    assert l % BLOCK_M == 0
    assert l % BLOCK_N == 0
    q = q.view(b, h, -1, BLOCK_M, d)
    k = k.view(b, h, -1, BLOCK_N, d)
    # calculate similarity within each block
    k_block_sim_mean = torch.einsum('bhlmd, bhlnd->bhlmn', k, k).mean(dim=(-2, -1))
    q_block_sim_mean = torch.einsum('bhlmd, bhlnd->bhlmn', q, q).mean(dim=(-2, -1))
    # get similarity mask for blocks with high similarity
    k_block_sim_mask = (k_block_sim_mean > simthreshd1).unsqueeze(-2).expand(1, 1, l//BLOCK_M, 1)
    q_block_sim_mask = (q_block_sim_mean > simthreshd1).unsqueeze(-1).expand(1, 1, 1, l//BLOCK_N)
    # calculate the 
    q_pooled = q.mean(dim=-2)
    k_pooled = k.mean(dim=-2)
    pre_softmax_sim = torch.einsum('bhlm,bhnm->bhln', q_pooled, k_pooled)
    pre_softmax_sim[~k_block_sim_mask] = -torch.inf
    pooled_score = torch.softmax(pre_softmax_sim, dim=-1)
    sorted_score = torch.sort(pooled_score, dim=-1, descending=True)
    cdf = torch.cumsum(sorted_score.values, dim=-1)
    B, H, Q, K = cdf.shape
    cdfthreshd_ts = cdfthreshd.view(1, H, 1, 1)
    cdfthreshd_ts = cdfthreshd_ts.expand(B, -1, Q, 1).contiguous()
    num_to_select = torch.searchsorted(cdf, cdfthreshd_ts, right=True).squeeze(-1)
    final_map = torch.zeros_like(pooled_score, dtype=torch.bool)
    final_map[~k_block_sim_mask] = 1
    final_map[~q_block_sim_mask] = 1
    return final_map
    
def selective_tile_attention(q, k, v, simthreshd1=0.1, cdfthreshd=0.9, return_sparsity=False):
    '''
    q: (batch, head, seq_len, d)
    k: (batch, head, seq_len, d)
    v: (batch, head, seq_len, d)
    simthreshd1: float
    cdfthreshd: float
    return_sparsity: bool
    '''
    block_sparse_mask = get_block_sparse_mask(q, k, simthreshd1, cdfthreshd, return_sparsity)
    pass