import torch

def generate_block_sparse_mask_for_function(h, num_q_blocks, num_kv_blocks, k, device="cuda"):
    """
    Generate block sparse mask of shape [h, num_q_blocks, num_kv_blocks].
    
    Args:
        h: number of heads
        num_q_blocks: number of query blocks
        num_kv_blocks: number of key/value blocks
        k: number of kv blocks each q block attends to
        device: device to create tensors on

    Returns:
        block_sparse_mask: [h, num_q_blocks, num_kv_blocks] bool tensor
    """
    k = min(k, num_kv_blocks)
    scores = torch.rand(h, num_q_blocks, num_kv_blocks, device=device)
    _, indices = torch.topk(scores, k, dim=-1)
    block_sparse_mask = torch.zeros(h, num_q_blocks, num_kv_blocks, dtype=torch.bool, device=device)

    block_sparse_mask = block_sparse_mask.scatter_(2, indices, 1).bool()
    return block_sparse_mask


def create_full_mask_from_block_mask(block_sparse_mask, q_variable_block_sizes, 
                                     kv_variable_block_sizes, device="cuda"):
    """
    Convert block-level sparse mask to full attention mask.
    
    Args:
        block_sparse_mask: [h, num_q_blocks, num_kv_blocks] bool tensor
        q_variable_block_sizes: [num_q_blocks] tensor
        kv_variable_block_sizes: [num_kv_blocks] tensor
        device: device to create tensors on

    Returns:
        full_mask: [h, S_q, S_kv] bool tensor where S = total sequence length
    """
    h, num_q_blocks, num_kv_blocks = block_sparse_mask.shape
    total_q_seq_len = q_variable_block_sizes.sum().item()
    total_kv_seq_len = kv_variable_block_sizes.sum().item()
    
    q_cumsum = torch.cat([torch.tensor([0], device=device), q_variable_block_sizes.cumsum(dim=0)[:-1]])
    kv_cumsum = torch.cat([torch.tensor([0], device=device), kv_variable_block_sizes.cumsum(dim=0)[:-1]])

    full_mask = torch.zeros(h, total_q_seq_len, total_kv_seq_len, dtype=torch.bool, device=device)

    for head in range(h):
        for q_block in range(num_q_blocks):
            q_start = q_cumsum[q_block]
            q_end = q_start + q_variable_block_sizes[q_block]

            for kv_block in range(num_kv_blocks):
                if block_sparse_mask[head, q_block, kv_block]:
                    kv_start = kv_cumsum[kv_block]
                    kv_end = kv_start + kv_variable_block_sizes[kv_block]
                    full_mask[head, q_start:q_end, kv_start:kv_end] = True

    return full_mask