# SPDX-License-Identifier: Apache-2.0
"""The negative prompt PAN2 was released with (``inference/prompts/negative_prompt.json`` in ifm-ai/PAN2).

Its ``time_range_sec`` fields are a ``0 to 10`` template; ``render_pan2_default_negative_prompt`` sets them to span
the clip, as the reference sampler does for records without a negative prompt of their own.
"""

import json

PAN2_DEFAULT_NEGATIVE_PROMPT = {
    "video_style": "low quality, blur, softness, reduced detail, noise, compression, clipped highlights, "
    "blurred sidebar, black border, letterboxing, watermark, subtitle, AIGC text, oily waxy "
    "plastic texture, oversaturation or color cast, color clipping or chroma bleeding, flicker, "
    "jitter, ghosting, motion smear, choppy or laggy motion, slow motion, Ken Burns effect, "
    "unmotivated slow zoom over a frozen scene, pan across a still image, global frame scaling "
    "or drifting, simulated movement from a static photograph, repeated micro-motion loop, "
    "discontinuous shot, scene cut, hard cut, crossfade, fade-in, fade-out, fading transition, "
    "scene replacement, spontaneous object or person pop-in or disappearance, duplicated or "
    "missing people or props, identity or count drift, rigid-object deformation or morphing, "
    "bending or melting rigid structures, structural shape or component drift",
    "lighting": "overexposure, clipped highlights, underexposure, harsh lighting, overly deep skin shadows, "
    "brightness fading, excessive lighting saturation, chroma-clipped highlights, neon-like skin "
    "tones, colored spill overwhelming local material colors, oversaturated bloom, unnatural neon "
    "color casts, hue drift, chroma noise, bad white balance, color pulsing, flickering light, "
    "unnatural lighting",
    "perspective_view": "warped geometry, distorted perspective, inconsistent spatial relationships, "
    "unmotivated layout changes, camera teleport or jump, viewpoint reset, wrong depth or "
    "occlusion",
    "subjects_lifecycle": [
        {
            "subject_id": "<|Subject1|>",
            "display_name": "malformed inconsistent subject",
            "subject_attribute": "broken anatomy, distorted body proportions, reversed pelvis, "
            "twisted torso, deformed arms or legs, reversed joints, fused "
            "feet or toes, warped or collapsing faces, transparent or "
            "floating head and body, missing body parts, detached or "
            "disembodied body parts, floating hands or limbs, detached "
            "articulated parts, disconnected joints or supports, hands "
            "disconnected from wrists or forearms, discontinuous "
            "limb-to-torso connections, blurred or fused fingers, missing "
            "thumbs, extra or missing fingers, malformed wrists, extra "
            "limbs, duplicated hands, mirrored face, identity drift, wrong "
            "relative sizes, human-object fusion, unintended body-material "
            "fusion, loss of separate material boundaries, opaque white "
            "nail polish, gray nail polish, chalky opaque fingernails, "
            "artificial acrylic nails",
        }
    ],
    "init_frame_visual_description": "blur or softness, reduced detail, noise or compression artifacts, clipped "
    "highlights, blurred sidebar, black border or letterboxing, oily waxy "
    "plastic surfaces, oversaturation or color cast, exaggerated local "
    "contrast, exposure fading, bad white balance, smeared dense textures, "
    "blurred or distorted face, transparent or missing body parts, detached or "
    "disembodied body parts, floating hands or limbs, detached articulated "
    "parts, disconnected joints or supports, disconnected wrists or forearms, "
    "discontinuous limb-to-torso connections, twisted torso, fused or missing "
    "fingers, missing thumbs, malformed wrists, duplicated limbs, deformed "
    "arms or legs, reversed knees, fused toes, warped geometry, deformed or "
    "asymmetrically warped rigid objects, bent rigid structures, incorrect "
    "rigid part geometry, topology, proportions, or component count, wrong "
    "occlusion, floating or hovering objects, unsupported objects, object or "
    "human-object fusion, unintended fusion of distinct surfaces or layers, "
    "loss of separate material boundaries, merged objects, hand-object "
    "penetration, object-object penetration, penetration, collision, "
    "impossible contact, wrong sizes.",
    "events": [
        {
            "event_index": 0,
            "time_range_sec": "0 to 10",
            "event_description": "blur, softness, reduced detail, compression artifacts, clipped "
            "highlights, flicker, jitter, ghosting, motion smear, choppy or laggy "
            "motion, slow motion, Ken Burns effect, unmotivated slow zoom over a "
            "frozen scene, pan across a still image, global frame scaling or drifting, "
            "simulated movement from a static photograph, repeated micro-motion loop, "
            "oily oversaturated or color-pulsing surfaces, shimmering surface detail, "
            "fingers fusing or changing count, hands smearing, facial features "
            "merging, reversed joints, upper body rotating independently of attached "
            "hips and feet, corkscrewed waist, limbs changing length or attachment "
            "after occlusion, transparent body or head, detached or disembodied body "
            "parts, floating hands or limbs, detached articulated parts, disconnected "
            "joints or supports, hands disconnected from wrists or forearms, body "
            "parts moving independently, vanishing limb-to-torso connections, merged "
            "people or limbs, limbs fused into torso, merged or fused feet and toes, "
            "floating or hovering objects, unsupported objects, object or human-object "
            "fusion, unintended fusion of distinct surfaces or layers, loss of "
            "separate material boundaries, merged objects, hand-object penetration, "
            "object-object penetration, penetration or clipping, objects colliding or "
            "morphing, rigid objects tearing, fragmenting, fusing, bending, melting or "
            "changing dimensions, topology, silhouette, component count or layout, "
            "subjects changing identity, splitting or duplicating, objects appearing "
            "or disappearing without entry, exit or occlusion, changed identity or "
            "count after occlusion, state changes inconsistent with the causing "
            "action, broken physics, uncaused deformation or motion, state changes "
            "preceding the causing interaction, loss of coupling during contact, pose "
            "jumps, discontinuous shot, camera jump, hard scene cut, crossfade, "
            "fade-in, fade-out, fading transition, scene replacement or viewpoint "
            "reset.",
            "camera_motion": [
                {"time_range_sec": "0 to 10", "camera_motion": ["camera_jump", "hard_cut", "viewpoint_reset"]}
            ],
            "contained_actions": [
                {
                    "time_range_sec": "0 to 10",
                    "action": "<|Subject1|> fingers fusing during grip changes, palms "
                    "merging with props, wrists collapsing, crossed limbs fusing, "
                    "intersecting or swapping attachment, duplicated hands or "
                    "feet, detached or disembodied hands or limbs, hands "
                    "disconnected from wrists or forearms, discontinuous "
                    "limb-to-torso connections, merged or fused feet and toes, "
                    "unstable contact, unintended grip slip, hand-object "
                    "penetration, body-object penetration, passage through solid "
                    "surfaces instead of openings, object-object penetration, "
                    "merged or fused objects, rigid objects fragmenting, fusing, "
                    "smearing or changing dimensions during motion, unintended "
                    "fusion of distinct surfaces or layers, loss of separate "
                    "material boundaries, floating or hovering objects, "
                    "unsupported objects, objects duplicating, splitting or "
                    "disappearing during interaction or occlusion, unchanged "
                    "action result, pose jump",
                }
            ],
        }
    ],
    "quality": {
        "video_quality": "low resolution, blur, softness, reduced detail, noise, compression, clipped "
        "highlights, blurred sidebar, black border, letterboxing, oily waxy plastic "
        "surfaces, greasy highlights, oversaturated colors, faded or washed-out "
        "colors, exposure fading, smeared faces, hands or held objects, surface detail "
        "smearing or swimming, material or existing markings changing without a "
        "physical cause, color or shape morphing, object pop-in or pop-out, sudden "
        "disappearance, missing persistent objects, object duplication or count drift, "
        "rigid-object warping or deformation, changing rigid proportions, topology, "
        "silhouette, or component layout",
        "text_logos_overlays": "watermark, subtitle, captions, AIGC text or logo",
        "artifacts": [
            "watermark",
            "template_or_text_dominated",
            "talking_head",
            "visual_quality_issue",
            "edited_or_discontinuous",
            "video_game",
            "animation",
            "ai_generated_content",
            "black_border",
            "ken_burn",
            "time_lapse",
            "blurred_sidebar",
            "hard_scene_cuts",
        ],
    },
}


def render_pan2_default_negative_prompt(num_frames: int, fps: float) -> str:
    """The default negative prompt as compact one-line JSON, with every time range spanning the clip."""
    span = f"0 to {(num_frames - 1) / fps:g}"

    def fill(node):
        if isinstance(node, dict):
            return {key: span if key == "time_range_sec" else fill(value) for key, value in node.items()}
        if isinstance(node, list):
            return [fill(item) for item in node]
        return node

    return json.dumps(fill(PAN2_DEFAULT_NEGATIVE_PROMPT), ensure_ascii=False, separators=(",", ":"))
