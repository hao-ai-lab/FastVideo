import torch

from fastvideo.utils.parallel_states import nccl_info


class RoPE1D:

    def __init__(self, freq=1e4, F0=1.0, scaling_factor=1.0):
        self.base = freq
        self.F0 = F0
        self.scaling_factor = scaling_factor
        self.cache = {}

    def get_cos_sin(self, D, seq_len, device, dtype):
        if (D, seq_len, device, dtype) not in self.cache:
            inv_freq = 1.0 / (self.base**(torch.arange(0, D, 2).float().to(device) / D))
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
            freqs = torch.cat((freqs, freqs), dim=-1)
            cos = freqs.cos()  # (Seq, Dim)
            sin = freqs.sin()
            self.cache[D, seq_len, device, dtype] = (cos, sin)
        return self.cache[D, seq_len, device, dtype]

    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope1d(self, tokens, pos1d, cos, sin):
        assert pos1d.ndim == 2
        cos = torch.nn.functional.embedding(pos1d, cos)[:, :, None, :]
        sin = torch.nn.functional.embedding(pos1d, sin)[:, :, None, :]
        return (tokens * cos) + (self.rotate_half(tokens) * sin)

    def __call__(self, tokens, positions):
        """
        input:
            * tokens: batch_size x ntokens x nheads x dim
            * positions: batch_size x ntokens (t position of each token)
        output:
            * tokens after applying RoPE2D (batch_size x ntokens x nheads x dim)
        """
        D = tokens.size(3)
        assert positions.ndim == 2  # Batch, Seq
        cos, sin = self.get_cos_sin(D, int(positions.max()) + 1, tokens.device, tokens.dtype)
        tokens = self.apply_rope1d(tokens, positions, cos, sin)
        return tokens


class RoPE3D(RoPE1D):

    def __init__(self, freq=1e4, F0=1.0, scaling_factor=1.0):
        super(RoPE3D, self).__init__(freq, F0, scaling_factor)
        self.position_cache = {}

    def get_mesh_3d(self, rope_positions, bsz):
        f, h, w = rope_positions

        if f"{f}-{h}-{w}" not in self.position_cache:
            x = torch.arange(f, device='cpu')
            y = torch.arange(h, device='cpu')
            z = torch.arange(w, device='cpu')
            self.position_cache[f"{f}-{h}-{w}"] = torch.cartesian_prod(x, y, z).view(1, f * h * w, 3).expand(bsz, -1, 3)
        return self.position_cache[f"{f}-{h}-{w}"]

    def __call__(self, tokens, rope_positions, ch_split, parallel=False):
        """
        input:
            * tokens: batch_size x ntokens x nheads x dim
            * rope_positions: list of (f, h, w)
        output:
            * tokens after applying RoPE2D (batch_size x ntokens x nheads x dim)
        """
        assert sum(ch_split) == tokens.size(-1)

        mesh_grid = self.get_mesh_3d(rope_positions, bsz=tokens.shape[0])
        out = []
        for i, (D, x) in enumerate(zip(ch_split, torch.split(tokens, ch_split, dim=-1))):
            cos, sin = self.get_cos_sin(D, int(mesh_grid.max()) + 1, tokens.device, tokens.dtype)

            if parallel:
                mesh = torch.chunk(mesh_grid[:, :, i], nccl_info.sp_size, dim=1)[nccl_info.rank_within_group].clone()
            else:
                mesh = mesh_grid[:, :, i].clone()
            x = self.apply_rope1d(x, mesh.to(tokens.device), cos, sin)
            out.append(x)

        tokens = torch.cat(out, dim=-1)
        return tokens
