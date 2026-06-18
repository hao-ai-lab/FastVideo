"""AdapterDenoiseLoop — per-request LoRA / ControlNet over one resident base (design_v3 §9.19).

The adapter plane: a request selects adapters (``DiffusionParams.adapters``); the loop applies each
active adapter's velocity delta on top of the base DiT's prediction, then re-integrates. Many adapters
are served over ONE resident base; the active set lives in the request/``LoopState`` (never global), so
two requests using different adapters interleave without smearing. Isolated ``WanDenoiseLoop`` subclass
(empty adapter set ⇒ identical to the base), so nothing else is touched.
"""
from __future__ import annotations

import numpy as np

from v2.loop.contracts import Done
from v2.platform import FLOW_MATCH_STEP
from v2.recipes.wan21.loop import WanDenoiseLoop


class AdapterDenoiseLoop(WanDenoiseLoop):
    def init(self, req, model, ctx):
        st = super().init(req, model, ctx)
        st.scratch["adapters"] = tuple(getattr(req.diffusion, "adapters", ()) or ())
        st.scratch["control"] = ctx.slots.get("control_signal")     # ControlNet conditioning (optional)
        return st

    def next(self, st):
        plan = super().next(st)
        adapters = st.scratch.get("adapters") or ()
        if isinstance(plan, Done) or not adapters:
            return plan
        base_run = plan.run
        x = st.latents["video"]
        control = st.scratch.get("control")
        sigma_t, sigma_next = st.sigmas[st.step_idx], st.sigmas[st.step_idx + 1]
        prec = self.precision

        def run(model, override=None):
            res = base_run(model, override)                         # base velocity (CFG combine)
            v = np.asarray(res.output["noise_pred"], dtype="float32")
            for aid in adapters:                                    # apply each active adapter's delta
                v = v + model.component(aid).delta(x, control)
            res.output["noise_pred"] = v
            fm = model.platform.kernels.get(FLOW_MATCH_STEP)    # solver dispatched per (device, arch)
            res.output["latents"] = fm(prec.cast(x), v, sigma_t, sigma_next).astype("float32")
            return res

        plan.run = run
        plan.label = f"{plan.label}.adapt[{'+'.join(adapters)}]"
        return plan
