"""Text-to-image → image-to-video: two distinct generation models chained by a cross-model
``Workflow`` (design_v3 §13). The realistic FLUX→Wan pipeline — two cards, two weight sets — proving
the design composes across model *instances*, not just loops within one (which LTX-2 already covers)."""
from __future__ import annotations

from .card import build_flux_t2i_card, build_wan_i2v_card
from .program import (
    build_flux_t2i_program,
    build_t2i_i2v_extend_workflow,
    build_t2i_then_i2v_workflow,
    build_wan_i2v_program,
)

__all__ = ["build_flux_t2i_card", "build_wan_i2v_card", "build_flux_t2i_program",
           "build_wan_i2v_program", "build_t2i_then_i2v_workflow", "build_t2i_i2v_extend_workflow"]
