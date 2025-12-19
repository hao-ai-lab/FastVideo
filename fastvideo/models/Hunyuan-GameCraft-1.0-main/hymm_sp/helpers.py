import torch
from typing import Union, List
from hymm_sp.modules.posemb_layers import get_1d_rotary_pos_embed, get_meshgrid_nd

from itertools import repeat
import collections.abc


def _ntuple(n):
    """
    Creates a helper function to convert inputs to tuples of specified length.
    
    Converts iterable inputs (excluding strings) to tuples of length n,
    or repeats single values n times to form a tuple. Useful for handling
    multi-dimensional parameters like sizes and strides.
    
    Args:
        n (int): Target length of the tuple
        
    Returns:
        function: Parser function that converts inputs to n-length tuples
    """
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            x = tuple(x)
            if len(x) == 1:
                x = tuple(repeat(x[0], n))
            return x
        return tuple(repeat(x, n))
    return parse


# Create common tuple conversion functions for 1-4 dimensions
to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)


def get_rope_freq_from_size(
    latents_size, 
    ndim, 
    target_ndim, 
    args,
    rope_theta_rescale_factor: Union[float, List[float]] = 1.0,
    rope_interpolation_factor: Union[float, List[float]] = 1.0,
    concat_dict={}
):
    """
    Calculates RoPE (Rotary Position Embedding) frequencies based on latent dimensions.
    
    Converts latent space dimensions to rope-compatible sizes by accounting for
    patch size, then generates the appropriate frequency embeddings for each dimension.
    
    Args:
        latents_size: Dimensions of the latent space tensor
        ndim (int): Number of dimensions in the latent space
        target_ndim (int): Target number of dimensions for the embeddings
        args: Configuration arguments containing model parameters (patch_size, rope_theta, etc.)
        rope_theta_rescale_factor: Rescaling factor(s) for theta parameter (per dimension)
        rope_interpolation_factor: Interpolation factor(s) for position embeddings (per dimension)
        concat_dict: Dictionary for special concatenation modes (e.g., time-based extensions)
    
    Returns:
        tuple: Cosine and sine frequency embeddings (freqs_cos, freqs_sin)
    """
    # Calculate rope sizes by dividing latent dimensions by patch size
    if isinstance(args.patch_size, int):
        # Validate all latent dimensions are divisible by patch size
        assert all(s % args.patch_size == 0 for s in latents_size), \
            f"Latent size (last {ndim} dimensions) must be divisible by patch size ({args.patch_size}), " \
            f"but got {latents_size}."
        rope_sizes = [s // args.patch_size for s in latents_size]
    elif isinstance(args.patch_size, list):
        # Validate with per-dimension patch sizes
        assert all(s % args.patch_size[idx] == 0 for idx, s in enumerate(latents_size)), \
            f"Latent size (last {ndim} dimensions) must be divisible by patch size ({args.patch_size}), " \
            f"but got {latents_size}."
        rope_sizes = [s // args.patch_size[idx] for idx, s in enumerate(latents_size)]

    # Add singleton dimensions if needed to match target_ndim (typically for time axis)
    if len(rope_sizes) != target_ndim:
        rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes

    # Calculate head dimension and validate rope dimensions
    head_dim = args.hidden_size // args.num_heads
    rope_dim_list = args.rope_dim_list
    
    # Default: split head dimension equally across target dimensions
    if rope_dim_list is None:
        rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
    
    # Ensure rope dimensions sum to head dimension
    assert sum(rope_dim_list) == head_dim, \
        "Sum of rope_dim_list must equal attention head dimension (hidden_size // num_heads)"

    # Generate rotary position embeddings
    freqs_cos, freqs_sin = get_nd_rotary_pos_embed_new(
        rope_dim_list, 
        rope_sizes, 
        theta=args.rope_theta, 
        use_real=True,
        theta_rescale_factor=rope_theta_rescale_factor,
        interpolation_factor=rope_interpolation_factor,
        concat_dict=concat_dict
    )
    return freqs_cos, freqs_sin


def get_nd_rotary_pos_embed_new(
    rope_dim_list, 
    start, 
    *args, 
    theta=10000., 
    use_real=False, 
    theta_rescale_factor: Union[float, List[float]] = 1.0,
    interpolation_factor: Union[float, List[float]] = 1.0,
    concat_dict={}
):
    """
    Generates multi-dimensional Rotary Position Embeddings (RoPE).
    
    Creates position embeddings for n-dimensional spaces by generating a meshgrid
    of positions and applying 1D rotary embeddings to each dimension, then combining them.
    
    Args:
        rope_dim_list (list): List of embedding dimensions for each axis
        start: Starting dimensions for generating the meshgrid
        *args: Additional arguments for meshgrid generation
        theta (float): Base theta parameter for RoPE frequency calculation
        use_real (bool): If True, returns separate cosine and sine embeddings
        theta_rescale_factor: Rescaling factor(s) for theta (per dimension)
        interpolation_factor: Interpolation factor(s) for position scaling (per dimension)
        concat_dict: Dictionary for special concatenation modes (e.g., time-based extensions)
    
    Returns:
        tuple or tensor: Cosine and sine embeddings if use_real=True, combined embedding otherwise
    """
    # Generate n-dimensional meshgrid of positions (shape: [dim, *sizes])
    grid = get_meshgrid_nd(start, *args, dim=len(rope_dim_list))
    
    # Handle special concatenation modes (e.g., adding time-based bias)
    if concat_dict:
        if concat_dict['mode'] == 'timecat':
            # Add bias as first element in first dimension
            bias = grid[:, :1].clone()
            bias[0] = concat_dict['bias'] * torch.ones_like(bias[0])
            grid = torch.cat([bias, grid], dim=1)
        elif concat_dict['mode'] == 'timecat-w':
            # Add biased first element with spatial offset
            bias = grid[:, :1].clone()
            bias[0] = concat_dict['bias'] * torch.ones_like(bias[0])
            bias[2] += start[-1]  # Spatial offset reference: OminiControl implementation
            grid = torch.cat([bias, grid], dim=1)

    # Normalize theta rescale factors to list format (per dimension)
    if isinstance(theta_rescale_factor, (int, float)):
        theta_rescale_factor = [theta_rescale_factor] * len(rope_dim_list)
    elif isinstance(theta_rescale_factor, list) and len(theta_rescale_factor) == 1:
        theta_rescale_factor = [theta_rescale_factor[0]] * len(rope_dim_list)
    assert len(theta_rescale_factor) == len(rope_dim_list), \
        "Length of theta_rescale_factor must match number of dimensions"

    # Normalize interpolation factors to list format (per dimension)
    if isinstance(interpolation_factor, (int, float)):
        interpolation_factor = [interpolation_factor] * len(rope_dim_list)
    elif isinstance(interpolation_factor, list) and len(interpolation_factor) == 1:
        interpolation_factor = [interpolation_factor[0]] * len(rope_dim_list)
    assert len(interpolation_factor) == len(rope_dim_list), \
        "Length of interpolation_factor must match number of dimensions"

    # Generate 1D rotary embeddings for each dimension and combine
    embs = []
    for i in range(len(rope_dim_list)):
        # Flatten grid dimension and generate embeddings
        emb = get_1d_rotary_pos_embed(
            rope_dim_list[i], 
            grid[i].reshape(-1),  # Flatten to 1D positions
            theta, 
            use_real=use_real,
            theta_rescale_factor=theta_rescale_factor[i],
            interpolation_factor=interpolation_factor[i]
        )
        embs.append(emb)

    # Combine embeddings from all dimensions
    if use_real:
        # Return separate cosine and sine components
        cos = torch.cat([emb[0] for emb in embs], dim=1)
        sin = torch.cat([emb[1] for emb in embs], dim=1)
        return cos, sin
    else:
        # Return combined embedding
        return torch.cat(embs, dim=1)
