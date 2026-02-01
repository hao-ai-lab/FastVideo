# SPDX-License-Identifier: Apache-2.0
# Muon optimizer - MomentUm Orthogonalized by Newton-schulz
# This code is modified from the GitHub repository of KellerJordan:
# https://github.com/KellerJordan/Muon/blob/master/muon.py

import math
import torch
try:
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor.placement_types import (
        Partial,
        Placement,
        Replicate,
        Shard,
    )
except ImportError:
    # handle old pytorch versions
    DTensor = None


def zeropower_via_newtonschulz5(G, steps=5):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    if DTensor is not None and isinstance(G, DTensor):
        device_mesh = G.device_mesh
        G = G.full_tensor()
    else:
        device_mesh = None

    assert len(G.shape) >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G
    if G.size(-2) > G.size(-1):
        X = X.mT
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT

    if device_mesh is not None:
        return DTensor.from_local(X, device_mesh)
    else:
        return X


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon is more memory-efficient than Adam because it only uses momentum (no second moment),
    reducing optimizer state memory by ~50%.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """

    def __init__(
            self,
            lr=1e-3,
            wd=0.1,
            muon_params=None,
            momentum=0.95,
            nesterov=True,
            ns_steps=5,
            adamw_params=None,
            adamw_betas=(0.95, 0.95),
            adamw_eps=1e-8,
    ):

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        # Sort parameters into those for which we will use Muon, and those for which we will not
        for p in muon_params:
            # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
            assert p.ndim >= 2, p.ndim
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            # Do not use Muon for parameters in adamw_params
            self.state[p]["use_muon"] = False

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as described in the paper
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]

            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

            # generate weight updates in distributed fashion
            for p in params:
                # sanity check
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None

                # calc update
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                g = g.bfloat16()
                u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

                # scale update
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

                # apply weight decay
                p.data.mul_(1 - lr * wd)

                # apply update
                p.data.add_(u.view(p.shape), alpha=-adjusted_lr)

            ############################
            #       AdamW backup       #
            ############################

            params = [
                p for p in group["params"] if not self.state[p]["use_muon"]
            ]
            lr = group['lr']
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss


def get_muon_optimizer(model,
                       lr=1e-3,
                       weight_decay=0.1,
                       momentum=0.95,
                       adamw_betas=(0.95, 0.95),
                       adamw_eps=1e-8):
    """
    Helper function to create a Muon optimizer for a model.

    Muon is used for parameters with ndim >= 2 (matrices),
    while AdamW is used for parameters with ndim < 2 (biases, norms, etc.)

    This is more memory-efficient than pure AdamW because Muon only uses
    momentum (no second moment), reducing optimizer state memory by ~50%.
    """
    muon_params = [
        p for name, p in model.named_parameters()
        if p.requires_grad and p.ndim >= 2
    ]
    adamw_params = [
        p for name, p in model.named_parameters()
        if p.requires_grad and not (p.ndim >= 2)
    ]

    return Muon(
        lr=lr,
        wd=weight_decay,
        muon_params=muon_params,
        momentum=momentum,
        adamw_params=adamw_params,
        adamw_betas=adamw_betas,
        adamw_eps=adamw_eps,
    )
