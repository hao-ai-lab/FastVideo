import os
import re
import time
from copy import deepcopy

import gradio as gr

from fastvideo.configs.sample.base import SamplingParam
from fastvideo.entrypoints.video_generator import VideoGenerator

from .config import (
    DEFAULT_FPS,
    DEFAULT_HEIGHT,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_NUM_FRAMES,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_WIDTH,
    MODEL_ID,
    MODEL_PATH_MAPPING,
    OUTPUT_DIR,
    setup_model_environment,
)
from .examples import load_example_prompts
from .prompt_rewrite import maybe_enhance_prompt
from .rendering import (
    create_timing_display,
    create_timing_placeholder,
    render_completed_clips,
    render_error_message,
    render_input_image_status,
    render_prompt_blocked_message,
    _record_session_clip,
)
from .safety import get_prompt_safety_check

def create_gradio_interface(default_params: dict[str, SamplingParam], generators: dict[str, VideoGenerator]):
    def _sanitize_filename_component(name: str) -> str:
        sanitized = re.sub(r'[\\/:*?"<>|]', "", name)
        sanitized = sanitized.strip().strip(".")
        sanitized = re.sub(r"\s+", "_", sanitized)
        return sanitized or "video"

    def generate_video(
        prompt, model_selection, input_image=None
    ):
        model_path = MODEL_PATH_MAPPING.get(model_selection, MODEL_ID)
        setup_model_environment(model_path)
        try:
            generator = generators[model_path]
            params = deepcopy(default_params[model_path])

            params.prompt = prompt
            params.seed = default_params[model_path].seed
            params.guidance_scale = default_params[model_path].guidance_scale
            params.num_frames = int(default_params[model_path].num_frames)
            params.height = int(default_params[model_path].height)
            params.width = int(default_params[model_path].width)
            params.fps = DEFAULT_FPS
            params.num_inference_steps = DEFAULT_NUM_INFERENCE_STEPS
            params.save_video = True
            params.return_frames = False
            params.output_path = ""
            params.negative_prompt = default_params[model_path].negative_prompt
            params.image_path = input_image or None

            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            safe_prompt = _sanitize_filename_component(prompt[:80])
            video_filename = f"{safe_prompt}_{int(time.time() * 1000)}.mp4"
            output_path = str(OUTPUT_DIR / video_filename)
            params.output_path = output_path
            start_time = time.perf_counter()
            result = generator.generate_video(
                prompt=prompt,
                output_path=output_path,
                fps=DEFAULT_FPS,
                seed=int(params.seed),
                save_video=True,
                return_frames=False,
                guidance_scale=float(params.guidance_scale),
                height=int(params.height),
                width=int(params.width),
                num_frames=int(params.num_frames),
                num_inference_steps=DEFAULT_NUM_INFERENCE_STEPS,
                negative_prompt=params.negative_prompt,
                image_path=params.image_path,
                ltx2_image_crf=0.0
            )
            wall_time = time.perf_counter() - start_time
            generation_time = (
                result.get("generation_time")
                if isinstance(result, dict) else None
            )
            e2e_latency = (
                result.get("e2e_latency")
                if isinstance(result, dict) else None
            )
            if generation_time is None:
                generation_time = wall_time
            if e2e_latency is None:
                e2e_latency = wall_time
            resolved_output_path = (
                result.get("output_path", output_path)
                if isinstance(result, dict) else output_path
            )
            logging_info = result.get("logging_info", None) if isinstance(result, dict) else None
            if logging_info:
                stage_names = logging_info.get_execution_order()
                stage_execution_times = [
                    logging_info.get_stage_info(stage_name).get("execution_time", 0.0) 
                    for stage_name in stage_names
                ]
            else:
                stage_names = []
                stage_execution_times = []

            return (
                resolved_output_path,
                params.seed,
                params.num_frames,
                float(generation_time),
                float(e2e_latency),
            )

        except Exception as e:
            print(f"An error occurred during local generation: {e}")
            return None, f"Generation failed: {str(e)}", 0, 0.0, 0.0

    examples, example_labels = load_example_prompts()
    curated_prompts = {
        prompt.strip() for prompt in examples if prompt.strip()
    }
    initial_example_label = None
    
    theme = gr.themes.Base().set(
        button_primary_background_fill="#2563eb",
        button_primary_background_fill_hover="#1d4ed8",
        button_primary_text_color="white",
        slider_color="#2563eb",
        checkbox_background_color_selected="#2563eb",
    )

    def get_default_values(model_name: str):
        model_path = MODEL_PATH_MAPPING.get(model_name)
        if model_path and model_path in default_params:
            params = default_params[model_path]
            return params.height, params.width, params.num_frames
        return DEFAULT_HEIGHT, DEFAULT_WIDTH, DEFAULT_NUM_FRAMES

    init_height, init_width, init_num_frames = get_default_values("FastLTX-2.3")

    def render_generation_badges(model_name: str) -> str:
        _ = model_name
        return """
        <div class="generation-badges">
            <span class="generation-badge">FastLTX-2.3</span>
            <span class="generation-badge">5 sec</span>
            <span class="generation-badge">1080p</span>
            <span class="generation-badge">9:16</span>
        </div>
        """

    with gr.Blocks(title="FastLTX-2.3", theme=theme) as demo:
        completed_clips_state = gr.State([])
        gr.HTML("""
        <div id="hero-shell" class="hero-shell">
            <div id="hero-brand" class="hero-brand">
                <img src="/logo.png" alt="FastVideo logo" id="hero-fastvideo-logo" class="hero-fastvideo-logo" />
                <img src="/nvidia.png" alt="NVIDIA logo" id="hero-nvidia-logo" class="hero-nvidia-logo" />
            </div>
            <div id="hero-title" class="hero-title">Real-Time 1080p Video Generation with FastLTX-2.3 on a single B200</div>
        </div>
        """, elem_id="hero-wrapper")

        with gr.Column(elem_id="app-shell", elem_classes="app-shell"):
            timing_title = gr.HTML(
                "<div class='timing-section-title'>TIMING BREAKDOWN</div>",
                visible=False,
                elem_id="timing-title",
            )
            timing_display = gr.Markdown(
                value=create_timing_placeholder(),
                visible=False,
                elem_id="timing-display",
                elem_classes="timing-display-block",
            )

            with gr.Group(elem_id="stage-card", elem_classes="stage-card"):
                with gr.Row(elem_id="stage-card-header", elem_classes="stage-card-header"):
                    gr.HTML(
                        "<div class='stage-title'>🏎️ Make Video Generation Go Blurrrrrrr 💨</div>"
                    )
                    stage_badges = gr.HTML(
                        render_generation_badges("FastLTX-2.3"),
                        elem_id="stage-badges",
                        elem_classes="stage-badges-wrap",
                    )

                result = gr.Video(
                    label="Generated Video",
                    show_label=False,
                    container=True,
                    visible=True,
                    elem_id="stage-video",
                    elem_classes="stage-video",
                )
                error_output = gr.HTML(visible=False, elem_id="error-output")

            with gr.Group(elem_id="control-card", elem_classes="control-card"):
                gr.HTML(
                    "<div class='control-field-label'>Select an example prompt below or create your own (and optionally add an input image as your first frame)</div>",
                    elem_id="example-dropdown-label",
                )
                example_dropdown = gr.Dropdown(
                    choices=example_labels,
                    show_label=False,
                    value=initial_example_label,
                    interactive=True,
                    allow_custom_value=False,
                    container=False,
                    elem_id="example-dropdown",
                )
                prompt_textbox = gr.Textbox(
                    show_label=False,
                    value="",
                    placeholder="Describe your scene...",
                    max_lines=3,
                    container=False,
                    lines=3,
                    autofocus=True,
                    elem_id="prompt-textbox",
                )

                model_selection = gr.Dropdown(
                    choices=list(MODEL_PATH_MAPPING.keys()),
                    value="FastLTX-2.3",
                    label="Model",
                    interactive=True,
                    visible=len(MODEL_PATH_MAPPING) > 1,
                    elem_id="model-selection",
                )
                input_image = gr.File(
                    show_label=False,
                    file_types=["image"],
                    type="filepath",
                    container=False,
                    elem_id="input-image",
                )
                with gr.Row(
                    elem_id="image-upload-status-row",
                    elem_classes="image-upload-status-row",
                ):
                    image_upload_status = gr.HTML(
                        value=render_input_image_status(None),
                        elem_id="image-upload-status",
                        elem_classes="image-upload-status-wrap",
                    )
                    clear_image_button = gr.Button(
                        "x",
                        variant="secondary",
                        size="sm",
                        visible=False,
                        elem_id="clear-image-button",
                    )

                with gr.Row(elem_id="control-footer-row", elem_classes="control-footer-row"):
                    with gr.Row(elem_id="control-actions-row", elem_classes="control-actions-row"):
                        gr.HTML(
                            """
                            <button
                                type="button"
                                class="upload-image-trigger"
                                aria-label="Upload image"
                                title="Upload image"
                                onclick="(() => { const input = document.querySelector('#input-image input[type=file]'); if (input) { input.value = ''; input.click(); } })()"
                            >
                                <svg
                                    viewBox="0 0 24 24"
                                    fill="none"
                                    xmlns="http://www.w3.org/2000/svg"
                                    aria-hidden="true"
                                >
                                    <path
                                        d="M5 6.5C5 5.67157 5.67157 5 6.5 5H9.2C9.59783 5 9.97936 4.84196 10.2607 4.56066L10.9393 3.88204C11.2206 3.60074 11.6022 3.4427 12 3.4427H13.8C14.1978 3.4427 14.5794 3.60074 14.8607 3.88204L15.5393 4.56066C15.8206 4.84196 16.2022 5 16.6 5H17.5C18.3284 5 19 5.67157 19 6.5V17.5C19 18.3284 18.3284 19 17.5 19H6.5C5.67157 19 5 18.3284 5 17.5V6.5Z"
                                        stroke="currentColor"
                                        stroke-width="1.7"
                                        stroke-linejoin="round"
                                    />
                                    <circle
                                        cx="12"
                                        cy="11.5"
                                        r="3"
                                        stroke="currentColor"
                                        stroke-width="1.7"
                                    />
                                    <path
                                        d="M7.25 16L9.9 13.35C10.2125 13.0375 10.7192 13.0375 11.0317 13.35L12.1 14.4183C12.4125 14.7308 12.9192 14.7308 13.2317 14.4183L14.95 12.7C15.2625 12.3875 15.7692 12.3875 16.0817 12.7L16.75 13.3683"
                                        stroke="currentColor"
                                        stroke-width="1.7"
                                        stroke-linecap="round"
                                        stroke-linejoin="round"
                                    />
                                </svg>
                            </button>
                            """,
                            elem_id="upload-image-trigger",
                        )
                        run_button = gr.Button(
                            "Create",
                            variant="primary",
                            size="lg",
                            elem_id="run-button",
                        )

                with gr.Row(visible=False):
                    height_display = gr.Number(
                        label="Height",
                        value=init_height,
                        interactive=False,
                        container=True,
                    )
                    width_display = gr.Number(
                        label="Width",
                        value=init_width,
                        interactive=False,
                        container=True,
                    )
                    num_frames_display = gr.Number(
                        label="Number of Frames",
                        value=init_num_frames,
                        interactive=False,
                        container=True,
                    )

        with gr.Row(elem_id="completed-clips-header-row", elem_classes="completed-clips-header-row"):
            with gr.Column(scale=4):
                gr.Markdown("## Gallery")
            with gr.Column(scale=1, min_width=170, elem_id="completed-clips-button-column", elem_classes="completed-clips-button-column"):
                clear_clips_button = gr.Button(
                    "Clear My Gallery",
                    variant="secondary",
                    size="sm",
                    min_width=140,
                    elem_id="clear-clips-button",
                )
        completed_clips_status = gr.Markdown(
            "Your completed clips for this browser session will appear here.",
            elem_id="completed-clips-status",
        )
        completed_clips_html = gr.HTML(
            value=render_completed_clips([]),
            elem_id="completed-clips-section",
            elem_classes="completed-clips-section",
        )

        gr.HTML("""
        <style>
        :root {
            --fv-bg: #000000;
            --fv-text: #f5f7fb;
            --fv-muted: #d3daea;
            --fv-border: rgba(68, 88, 128, 0.82);
            --fv-panel: #000000;
            --fv-panel-soft: #000000;
            --fv-chip: linear-gradient(180deg, rgba(10, 19, 38, 0.92), rgba(5, 10, 20, 0.92));
            --fv-surface: rgba(6, 11, 22, 0.94);
            --fv-surface-soft: rgba(8, 12, 24, 0.92);
            --fv-shadow: rgba(0, 0, 0, 0.28);
            --fv-overscroll-shift: 0px;
            --body-background-fill: #000000;
            --body-background-fill-subdued: #000000;
            --background-fill-primary: #000000;
            --background-fill-secondary: #000000;
            --block-background-fill: #000000;
            --block-background-fill-dark: #000000;
            --panel-background-fill: #000000;
            --panel-background-fill-dark: #000000;
            --input-background-fill: #020713;
            --input-background-fill-focus: #020713;
            --fv-hero-side-width: 220px;
            --body-text-color: var(--fv-text) !important;
            --body-text-color-subdued: var(--fv-muted) !important;
        }

        html,
        body,
        #root,
        .gradio-container,
        .main,
        .app {
            background: var(--fv-panel) !important;
            background-color: var(--fv-panel) !important;
            background-image: none !important;
        }

        html {
            background: #000000 !important;
            background-color: #000000 !important;
            background-image: none !important;
            scroll-behavior: smooth;
            overscroll-behavior-y: auto;
        }

        body {
            min-height: 100%;
            width: 100vw !important;
            overflow-x: hidden !important;
            overflow-y: auto !important;
            -webkit-overflow-scrolling: touch;
            background: #000000 !important;
            background-color: #000000 !important;
            background-image: none !important;
            position: relative;
        }

        body::before {
            content: "";
            position: fixed;
            inset: 0;
            background: #000000;
            pointer-events: none;
            z-index: -1;
        }

        .gradio-container {
            width: 100% !important;
            max-width: 100% !important;
            margin: 0 auto !important;
            padding: 14px 18px 32px !important;
            background: var(--fv-panel) !important;
            position: relative;
            isolation: isolate;
        }

        #hero-wrapper,
        #hero-shell,
        #timing-title,
        #timing-display,
        .completed-clips-header-row,
        #completed-clips-status,
        #completed-clips-section {
            transform: translate3d(0, var(--fv-overscroll-shift), 0);
            transition: transform 240ms cubic-bezier(0.22, 1, 0.36, 1);
            will-change: transform;
        }

        .main {
            width: 100% !important;
            max-width: 100% !important;
            margin: 0 auto !important;
            background: var(--fv-panel) !important;
        }

        footer {
            display: none !important;
        }

        .gradio-container::before,
        .gradio-container::after {
            display: none !important;
        }

        .gr-block,
        .gr-form,
        .gr-box,
        .gr-group,
        .gr-panel,
        .block {
            background: transparent !important;
            box-shadow: none !important;
        }

        #root,
        #root > .app,
        #root .main,
        .gradio-container > .main,
        .gradio-container > .main > div,
        .gradio-container > div,
        .contain {
            width: 100% !important;
            max-width: 100% !important;
            background: var(--fv-panel) !important;
            background-color: var(--fv-panel) !important;
            background-image: none !important;
        }

        #hero-wrapper {
            width: 100% !important;
            background: transparent !important;
        }

        #hero-shell {
            display: grid;
            grid-template-columns: var(--fv-hero-side-width) minmax(0, 1fr) var(--fv-hero-side-width);
            align-items: center;
            gap: 18px;
            width: min(1320px, calc(100vw - 72px));
            max-width: 100%;
            min-height: 76px;
            margin: 0 auto 10px;
            padding: 12px 24px;
            border-radius: 22px;
            border: 1px solid var(--fv-border);
            background:
                radial-gradient(circle at center, rgba(24, 81, 200, 0.14), transparent 42%),
                linear-gradient(180deg, rgba(9, 14, 24, 0.94), rgba(4, 7, 14, 0.9));
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.04),
                0 16px 38px var(--fv-shadow);
        }

        #hero-shell::after {
            content: "";
            width: var(--fv-hero-side-width);
        }

        #hero-brand {
            display: flex;
            align-items: center;
            justify-content: flex-start;
            width: var(--fv-hero-side-width);
        }

        #hero-fastvideo-logo {
            height: 44px;
            width: auto;
        }

        #hero-nvidia-logo {
            height: 34px;
            width: auto;
        }

        #hero-title {
            grid-column: 2;
            color: var(--fv-text) !important;
            text-align: center;
            font-size: 1rem;
            font-weight: 850;
            line-height: 1.15;
            letter-spacing: 0.01em;
        }

        #app-shell {
            max-width: 900px;
            margin: 0 auto;
            gap: 12px;
            position: relative;
            z-index: 1;
        }

        #timing-title .timing-section-title {
            color: var(--fv-text) !important;
            text-align: center;
            font-size: 0.9rem;
            font-weight: 900;
            letter-spacing: 0.08em;
            margin-bottom: 8px;
        }

        #timing-display {
            margin-bottom: 14px !important;
        }

        #stage-card,
        #control-card {
            border-radius: 22px !important;
            border: 1px solid var(--fv-border) !important;
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.03),
                0 16px 38px var(--fv-shadow) !important;
            background: #000000 !important;
        }

        #stage-card {
            padding: 14px !important;
            margin-bottom: 0 !important;
        }

        #control-card {
            padding: 10px !important;
            margin-top: 12px !important;
            margin-bottom: 34px !important;
            position: relative !important;
            z-index: 20 !important;
            overflow: visible !important;
        }

        #stage-card > div,
        #stage-card > .gap,
        #stage-card .gap,
        #stage-card .gr-form,
        #stage-card .gr-box,
        #stage-card .gr-group,
        #stage-card .gr-panel,
        #stage-card .block,
        #stage-card .wrap,
        #control-card > div,
        #control-card > .gap,
        #control-card .gap,
        #control-card .gr-form,
        #control-card .gr-box,
        #control-card .gr-group,
        #control-card .gr-panel,
        #control-card .block,
        #control-card .wrap {
            background: transparent !important;
            box-shadow: none !important;
        }

        #stage-card-header {
            align-items: center !important;
            justify-content: space-between !important;
            gap: 12px !important;
            margin-bottom: 10px !important;
        }

        #stage-card .stage-title {
            color: var(--fv-text) !important;
            font-size: 1.02rem;
            font-weight: 900;
            line-height: 1.25;
        }

        #stage-badges {
            margin: 0 !important;
        }

        .generation-badges {
            display: flex;
            flex-wrap: wrap;
            justify-content: flex-end;
            gap: 8px;
        }

        .generation-badge {
            display: inline-flex;
            align-items: center;
            padding: 6px 10px;
            border-radius: 999px;
            border: 1px solid rgba(102, 122, 160, 0.48);
            background: var(--fv-chip);
            color: var(--fv-muted);
            font-size: 0.78rem;
            font-weight: 700;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
        }

        #stage-video,
        #stage-video .wrap,
        #stage-video video {
            width: 100% !important;
        }

        #stage-video {
            margin: 0 !important;
            padding: 14px !important;
            border-radius: 22px !important;
            overflow: hidden !important;
            background: #04070d !important;
            box-shadow: inset 0 0 0 1px rgba(37, 99, 235, 0.12) !important;
        }

        #stage-video .wrap {
            position: relative !important;
            border-radius: 18px !important;
            overflow: hidden !important;
            margin: 0 !important;
            padding: 0 !important;
            line-height: 0 !important;
            box-shadow: inset 0 0 0 1px rgba(37, 99, 235, 0.1) !important;
            background:
                radial-gradient(circle at center, rgba(18, 57, 140, 0.16), transparent 32%),
                #050913 !important;
        }

        #stage-video .wrap .progress-text,
        #stage-video .wrap .meta-text {
            line-height: 1.2 !important;
            height: auto !important;
        }

        #stage-video video {
            display: block !important;
            width: 100% !important;
            height: auto !important;
            max-height: none !important;
            border-radius: 18px !important;
            object-fit: contain !important;
            overflow: hidden !important;
            background:
                radial-gradient(circle at center, rgba(18, 57, 140, 0.16), transparent 32%),
                #050913 !important;
        }

        #stage-video .download-link,
        #stage-video .download-button,
        #stage-video a[download],
        #stage-video button[aria-label*="download" i],
        #stage-video [title*="download" i] {
            display: none !important;
        }

        #error-output {
            margin-top: 10px !important;
        }

        .stage-error-card {
            border-radius: 18px;
            border: 1px solid rgba(248, 113, 113, 0.35);
            background:
                linear-gradient(180deg, rgba(45, 12, 18, 0.96), rgba(24, 8, 13, 0.96));
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.03),
                0 10px 28px rgba(0, 0, 0, 0.18);
            padding: 14px 16px;
            color: #fecaca;
        }

        .stage-error-title {
            color: #fee2e2;
            font-size: 0.82rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }

        .stage-error-copy {
            color: #fca5a5;
            font-size: 0.94rem;
            line-height: 1.55;
        }

        #control-card-title {
            margin-bottom: 8px !important;
        }

        #example-dropdown-label {
            margin-bottom: 6px !important;
            background: transparent !important;
        }

        .control-field-label {
            color: var(--fv-text) !important;
            font-size: 0.92rem;
            font-weight: 700;
            line-height: 1.3;
            padding: 0 2px;
        }

        #control-card .gr-form,
        #control-card .gr-group {
            background: transparent !important;
            box-shadow: none !important;
        }

        #example-dropdown,
        #model-selection,
        #input-image {
            margin-bottom: 10px !important;
            background: transparent !important;
        }

        #example-dropdown,
        #model-selection {
            position: relative !important;
            z-index: 21 !important;
        }

        #example-dropdown .block,
        #example-dropdown .gr-form,
        #example-dropdown .gr-box,
        #example-dropdown .gr-group,
        #example-dropdown .gr-panel {
            background: transparent !important;
            box-shadow: none !important;
        }

        #example-dropdown .wrap,
        #model-selection .wrap {
            background:
                linear-gradient(180deg, rgba(10, 16, 31, 0.94), rgba(5, 9, 18, 0.94)) !important;
            border: 1px solid rgba(62, 78, 108, 0.72) !important;
            border-radius: 14px !important;
        }

        #example-dropdown [role="listbox"],
        #example-dropdown ul,
        #example-dropdown .options {
            max-height: 220px !important;
            overflow-y: auto !important;
            overscroll-behavior: contain !important;
            background: #050913 !important;
        }

        #prompt-textbox {
            margin-bottom: 10px !important;
        }

        #prompt-textbox,
        #prompt-textbox > div,
        #prompt-textbox .block,
        #prompt-textbox .gr-form,
        #prompt-textbox .gr-box,
        #prompt-textbox .gr-group,
        #prompt-textbox .gr-panel,
        #prompt-textbox .wrap {
            background: transparent !important;
            box-shadow: none !important;
        }

        #prompt-textbox textarea {
            min-height: 72px !important;
            max-height: 72px !important;
            padding: 12px 14px !important;
            border: 1px solid rgba(62, 78, 108, 0.72) !important;
            border-radius: 14px !important;
            background:
                linear-gradient(180deg, rgba(8, 14, 28, 0.94), rgba(5, 9, 18, 0.94)) !important;
            color: var(--fv-text) !important;
            font-size: 0.92rem !important;
            font-weight: 400 !important;
            line-height: 1.45 !important;
            resize: none !important;
            overflow-y: auto !important;
            scrollbar-width: thin;
            scrollbar-color: rgba(91, 112, 154, 0.8) rgba(8, 14, 28, 0.3);
        }

        #prompt-textbox textarea::placeholder {
            color: rgba(211, 218, 234, 0.58) !important;
            font-weight: 500 !important;
        }

        #prompt-textbox textarea:focus {
            border-color: rgba(93, 124, 188, 0.86) !important;
            box-shadow: inset 0 0 0 1px rgba(46, 102, 255, 0.18) !important;
        }

        #prompt-textbox textarea::-webkit-scrollbar {
            width: 8px;
        }

        #prompt-textbox textarea::-webkit-scrollbar-track {
            background: rgba(8, 14, 28, 0.32);
            border-radius: 999px;
        }

        #prompt-textbox textarea::-webkit-scrollbar-thumb {
            background: rgba(91, 112, 154, 0.8);
            border-radius: 999px;
        }

        #control-card label {
            font-weight: 700 !important;
            color: var(--fv-text) !important;
        }

        #input-image {
            position: absolute !important;
            width: 1px !important;
            height: 1px !important;
            min-height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            opacity: 0 !important;
            overflow: hidden !important;
            pointer-events: none !important;
        }

        #input-image .wrap,
        #input-image label {
            margin: 0 !important;
            padding: 0 !important;
            border: 0 !important;
            min-height: 0 !important;
            background: transparent !important;
            box-shadow: none !important;
        }

        #control-footer-row {
            align-items: center !important;
            justify-content: center !important;
            gap: 12px !important;
            margin-top: 2px !important;
        }

        #control-actions-row {
            display: flex !important;
            flex-wrap: nowrap !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 8px !important;
            margin: 0 auto !important;
            width: fit-content !important;
            max-width: 100% !important;
        }

        #control-actions-row > * {
            flex: 0 0 auto !important;
            display: flex !important;
            align-items: center !important;
            align-self: center !important;
            width: auto !important;
            max-width: none !important;
            margin: 0 !important;
        }

        #upload-image-trigger {
            flex: 0 0 auto !important;
            display: flex !important;
            align-items: center !important;
            align-self: center !important;
            width: auto !important;
            margin: 0 !important;
        }

        #upload-image-trigger .upload-image-trigger {
            width: 52px;
            min-width: 52px;
            height: 46px;
            padding: 0;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 10px;
            border: 1px solid rgba(86, 105, 142, 0.6);
            background:
                linear-gradient(180deg, rgba(15, 24, 42, 0.96), rgba(10, 16, 29, 0.96));
            color: transparent !important;
            font-size: 0 !important;
            line-height: 0 !important;
            text-indent: -9999px;
            overflow: hidden;
            position: relative;
            cursor: pointer;
            transform: translateY(2px);
            transition: border-color 160ms ease, transform 160ms ease, background 160ms ease;
        }

        #upload-image-trigger .upload-image-trigger::before {
            content: "";
            width: 24px;
            height: 24px;
            position: absolute;
            left: 50%;
            top: 50%;
            transform: translate(-50%, -50%);
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none'%3E%3Cpath d='M4.75 5.25C4.75 4.69772 5.19772 4.25 5.75 4.25H10.1C10.3652 4.25 10.6196 4.35536 10.8071 4.54289L11.4571 5.19289C11.6446 5.38043 11.899 5.48579 12.1642 5.48579H18.25C18.8023 5.48579 19.25 5.9335 19.25 6.48579V18.25C19.25 18.8023 18.8023 19.25 18.25 19.25H5.75C5.19771 19.25 4.75 18.8023 4.75 18.25V5.25Z' stroke='%23F5F7FB' stroke-width='1.7' stroke-linejoin='round'/%3E%3Ccircle cx='9.25' cy='9.1' r='1.55' stroke='%23F5F7FB' stroke-width='1.7'/%3E%3Cpath d='M6.5 16.5L10.15 12.85C10.4625 12.5375 10.9692 12.5375 11.2817 12.85L12.15 13.7183C12.4625 14.0308 12.9692 14.0308 13.2817 13.7183L17.5 9.5' stroke='%23F5F7FB' stroke-width='1.7' stroke-linecap='round' stroke-linejoin='round'/%3E%3Cpath d='M16.75 4.5V8.25' stroke='%23F5F7FB' stroke-width='1.7' stroke-linecap='round'/%3E%3Cpath d='M14.875 6.375H18.625' stroke='%23F5F7FB' stroke-width='1.7' stroke-linecap='round'/%3E%3C/svg%3E");
            background-repeat: no-repeat;
            background-position: center;
            background-size: 24px 24px;
        }

        #upload-image-trigger .upload-image-trigger svg {
            display: none !important;
        }

        #upload-image-trigger .upload-image-trigger:hover {
            border-color: rgba(120, 144, 188, 0.78);
            transform: translateY(1px);
            background:
                linear-gradient(180deg, rgba(18, 30, 54, 0.98), rgba(11, 18, 34, 0.98));
        }

        #image-upload-status-row {
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            width: 100% !important;
            gap: 3px !important;
            margin: 0 0 10px !important;
        }

        #image-upload-status {
            min-height: 0 !important;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            flex: 0 0 auto !important;
            width: auto !important;
        }

        .image-upload-status {
            display: inline-flex;
            align-items: center;
            padding: 6px 10px;
            border-radius: 999px;
            border: 1px solid rgba(74, 94, 134, 0.52);
            background: rgba(7, 13, 24, 0.92);
            color: var(--fv-muted);
            font-size: 0.82rem;
            font-weight: 700;
            line-height: 1.2;
        }

        #clear-image-button {
            --button-secondary-background-fill: rgba(88, 16, 28, 0.24) !important;
            --button-secondary-background-fill-hover: rgba(120, 22, 38, 0.32) !important;
            --button-secondary-border-color: rgba(248, 113, 113, 0.94) !important;
            --button-secondary-border-color-hover: rgba(252, 165, 165, 1) !important;
            --button-secondary-text-color: #f87171 !important;
            flex: 0 0 auto !important;
            width: auto !important;
            min-width: 0 !important;
            margin-left: -1px !important;
            background: transparent !important;
        }

        #clear-image-button > div {
            background: transparent !important;
            border-radius: 999px !important;
        }

        #clear-image-button button::before,
        #clear-image-button button::after {
            display: none !important;
        }

        #clear-image-button button {
            min-width: 28px !important;
            width: 28px !important;
            height: 28px !important;
            min-height: 28px !important;
            padding: 0 !important;
            border-radius: 999px !important;
            border: 1.5px solid rgba(248, 113, 113, 0.94) !important;
            background-color: rgba(88, 16, 28, 0.24) !important;
            background:
                linear-gradient(180deg, rgba(120, 22, 38, 0.18), rgba(72, 12, 18, 0.28)) !important;
            color: #f87171 !important;
            font-size: 0.95rem !important;
            font-weight: 900 !important;
            line-height: 1 !important;
            text-transform: none !important;
            text-shadow: 0 0 10px rgba(248, 113, 113, 0.18) !important;
            box-shadow:
                inset 0 0 0 1px rgba(248, 113, 113, 0.22),
                inset 0 1px 0 rgba(255, 255, 255, 0.08),
                0 0 0 1px rgba(127, 29, 29, 0.18),
                0 8px 20px rgba(48, 7, 12, 0.18) !important;
            backdrop-filter: blur(18px) saturate(150%);
            -webkit-backdrop-filter: blur(18px) saturate(150%);
            transition:
                border-color 160ms ease,
                background 160ms ease,
                transform 160ms ease,
                color 160ms ease,
                box-shadow 160ms ease !important;
        }

        #clear-image-button button:hover {
            border-color: rgba(252, 165, 165, 1) !important;
            background-color: rgba(120, 22, 38, 0.32) !important;
            background:
                linear-gradient(180deg, rgba(148, 29, 50, 0.24), rgba(88, 16, 28, 0.36)) !important;
            color: #fca5a5 !important;
            transform: translateY(-1px);
            box-shadow:
                inset 0 0 0 1px rgba(252, 165, 165, 0.24),
                inset 0 1px 0 rgba(255, 255, 255, 0.1),
                0 10px 24px rgba(66, 9, 18, 0.24) !important;
        }

        #clear-image-button button:active {
            transform: translateY(0);
        }

        #run-button {
            --button-primary-background-fill: rgba(10, 16, 29, 0.96) !important;
            --button-primary-background-fill-hover: rgba(11, 18, 34, 0.98) !important;
            --button-primary-border-color: rgba(86, 105, 142, 0.6) !important;
            --button-primary-border-color-hover: rgba(120, 144, 188, 0.78) !important;
            --button-primary-text-color: #f5f7fb !important;
            flex: 0 0 auto !important;
            display: flex !important;
            align-items: center !important;
            align-self: center !important;
            width: 220px !important;
            max-width: 220px !important;
            min-width: 132px !important;
            height: 46px !important;
            border-radius: 10px !important;
            background: transparent !important;
            box-shadow: none !important;
            border: 0 !important;
            overflow: visible !important;
        }

        #run-button > div {
            background: transparent !important;
            border-radius: 10px !important;
            box-shadow: none !important;
        }

        #run-button::before,
        #run-button::after,
        #run-button button::before,
        #run-button button::after {
            display: none !important;
        }

        #run-button,
        #run-button button {
            width: 100% !important;
            min-width: 220px !important;
            max-width: 220px !important;
            height: 46px !important;
            min-height: 46px !important;
            padding: 0 18px !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            border-radius: 10px !important;
            border: 1px solid rgba(86, 105, 142, 0.6) !important;
            background-color: rgba(10, 16, 29, 0.96) !important;
            background:
                linear-gradient(180deg, rgba(15, 24, 42, 0.96), rgba(10, 16, 29, 0.96)) !important;
            color: #f5f7fb !important;
            font-weight: 700 !important;
            line-height: 1 !important;
            letter-spacing: 0 !important;
            cursor: pointer !important;
            transform: translateY(2px);
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.04),
                0 10px 24px rgba(0, 0, 0, 0.18) !important;
            backdrop-filter: blur(18px) saturate(155%);
            -webkit-backdrop-filter: blur(18px) saturate(155%);
            transition:
                border-color 160ms ease,
                transform 160ms ease,
                background 160ms ease,
                box-shadow 160ms ease !important;
        }

        #run-button:hover,
        #run-button button:hover {
            border-color: rgba(120, 144, 188, 0.78) !important;
            background-color: rgba(11, 18, 34, 0.98) !important;
            transform: translateY(1px);
            background:
                linear-gradient(180deg, rgba(18, 30, 54, 0.98), rgba(11, 18, 34, 0.98)) !important;
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.08),
                0 12px 28px rgba(8, 17, 38, 0.28) !important;
        }

        #run-button:active,
        #run-button button:active {
            transform: translateY(2px);
        }

        .timing-shell {
            margin: 0 0 8px !important;
        }

        .timing-card {
            background: rgba(5, 11, 18, 0.92) !important;
            border: 1px solid rgba(66, 83, 116, 0.68) !important;
            color: var(--fv-text) !important;
            padding: 14px 10px;
            border-radius: 18px;
            text-align: center;
            min-height: 80px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .timing-card-highlight {
            background: rgba(5, 11, 18, 0.98) !important;
            border: 1px solid rgba(21, 108, 255, 0.9) !important;
            box-shadow: inset 0 0 0 1px rgba(21, 108, 255, 0.24) !important;
        }

        .performance-card {
            background: rgba(5, 11, 18, 0.92) !important;
            border: 1px solid rgba(66, 83, 116, 0.68) !important;
            color: var(--fv-text) !important;
            padding: 14px 10px;
            border-radius: 18px;
            text-align: center;
        }

        .completed-clips-header-row {
            max-width: 900px !important;
            margin: 8px auto 6px !important;
            align-items: center !important;
        }

        #completed-clips-button-column {
            display: flex !important;
            justify-content: flex-end !important;
            align-items: center !important;
        }

        #clear-clips-button {
            --button-secondary-background-fill: rgba(10, 16, 29, 0.96) !important;
            --button-secondary-background-fill-hover: rgba(11, 18, 34, 0.98) !important;
            --button-secondary-border-color: rgba(86, 105, 142, 0.78) !important;
            --button-secondary-border-color-hover: rgba(120, 144, 188, 0.86) !important;
            --button-secondary-text-color: #f5f7fb !important;
            width: auto !important;
            background: transparent !important;
            border: 0 !important;
            box-shadow: none !important;
        }

        #completed-clips-button-column > *,
        #clear-clips-button > div {
            background: transparent !important;
            border-radius: 12px !important;
            box-shadow: none !important;
        }

        #clear-clips-button::before,
        #clear-clips-button::after,
        #clear-clips-button button::before,
        #clear-clips-button button::after {
            display: none !important;
        }

        #clear-clips-button,
        #clear-clips-button button,
        #completed-clips-button-column button {
            width: auto !important;
            min-width: 140px !important;
            height: 36px !important;
            min-height: 36px !important;
            padding: 0 14px !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            border-radius: 10px !important;
            border: 1px solid rgba(86, 105, 142, 0.6) !important;
            background-color: rgba(10, 16, 29, 0.96) !important;
            background:
                linear-gradient(180deg, rgba(15, 24, 42, 0.96), rgba(10, 16, 29, 0.96)) !important;
            color: #f5f7fb !important;
            font-weight: 700 !important;
            line-height: 1 !important;
            cursor: pointer !important;
            transform: translateY(2px);
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.04),
                0 10px 24px rgba(0, 0, 0, 0.18) !important;
            backdrop-filter: blur(18px) saturate(155%);
            -webkit-backdrop-filter: blur(18px) saturate(155%);
            transition:
                border-color 160ms ease,
                background 160ms ease,
                transform 160ms ease,
                box-shadow 160ms ease !important;
        }

        #clear-clips-button:hover,
        #clear-clips-button button:hover,
        #completed-clips-button-column button:hover {
            border-color: rgba(120, 144, 188, 0.78) !important;
            background-color: rgba(11, 18, 34, 0.98) !important;
            background:
                linear-gradient(180deg, rgba(18, 30, 54, 0.98), rgba(11, 18, 34, 0.98)) !important;
            transform: translateY(1px);
            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.08),
                0 12px 28px rgba(8, 17, 38, 0.28) !important;
        }

        #clear-clips-button:active,
        #clear-clips-button button:active,
        #completed-clips-button-column button:active {
            transform: translateY(2px);
        }

        .completed-clips-header-row h2 {
            margin: 0 !important;
            color: var(--fv-text) !important;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-size: 0.95rem !important;
        }

        #completed-clips-status,
        #completed-clips-section {
            max-width: 900px;
            margin-left: auto;
            margin-right: auto;
        }

        #completed-clips-status {
            color: var(--fv-text) !important;
        }

        #completed-clips-section {
            border-radius: 20px;
            overflow-x: auto;
            overflow-y: hidden;
            padding-bottom: 10px;
            scrollbar-width: thin;
            scrollbar-color: rgba(91, 112, 154, 0.8) rgba(8, 14, 28, 0.3);
        }

        .completed-clips-grid {
            display: flex;
            flex-wrap: nowrap;
            align-items: stretch;
            gap: 18px;
            margin-top: 4px;
            width: max-content;
            min-width: 100%;
            padding-bottom: 2px;
        }

        #completed-clips-section::-webkit-scrollbar {
            height: 10px;
        }

        #completed-clips-section::-webkit-scrollbar-track {
            background: rgba(8, 14, 28, 0.32);
            border-radius: 999px;
        }

        #completed-clips-section::-webkit-scrollbar-thumb {
            background: rgba(91, 112, 154, 0.8);
            border-radius: 999px;
        }

        .completed-clip-card {
            flex: 0 0 340px;
            width: 340px;
            background:
                radial-gradient(circle at top, rgba(37, 99, 235, 0.16), transparent 38%),
                linear-gradient(180deg, rgba(7, 16, 31, 0.98), rgba(5, 10, 20, 0.98));
            border: 1px solid rgba(96, 165, 250, 0.28);
            border-radius: 22px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
            overflow: hidden;
            padding: 14px;
        }

        .completed-clip-video-shell {
            background: rgba(15, 23, 42, 0.82);
            border: 1px solid rgba(148, 163, 184, 0.24);
            border-radius: 18px;
            overflow: hidden;
        }

        .completed-clip-video {
            display: block;
            width: 100%;
            aspect-ratio: 16 / 9;
            object-fit: cover;
            background: #020617;
        }

        .completed-clip-body {
            padding: 14px 2px 2px;
        }

        .completed-clip-title {
            color: #f8fafc;
            font-size: 1.02rem;
            font-weight: 700;
            line-height: 1.35;
            margin-bottom: 12px;
        }

        .completed-clip-meta {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 8px;
            margin-bottom: 12px;
        }

        .completed-clip-badge {
            display: inline-flex;
            align-items: center;
            padding: 6px 10px;
            border-radius: 999px;
            border: 1px solid rgba(148, 163, 184, 0.26);
            background: rgba(15, 23, 42, 0.9);
            color: #dbeafe;
            font-size: 0.82rem;
            font-weight: 600;
        }

        .completed-clip-duration {
            background: rgba(30, 41, 59, 0.94);
            color: #e2e8f0;
        }

        .completed-clip-prompt {
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 14px;
            background: rgba(8, 15, 29, 0.96);
            overflow: hidden;
        }

        .completed-clip-prompt summary {
            cursor: pointer;
            list-style: none;
            padding: 10px 14px;
            color: #f8fafc;
            font-weight: 600;
        }

        .completed-clip-prompt summary::-webkit-details-marker {
            display: none;
        }

        .completed-clip-prompt div {
            padding: 0 14px 14px;
            color: #cbd5e1;
            line-height: 1.5;
        }

        .completed-clips-empty {
            border: 1px dashed rgba(148, 163, 184, 0.28);
            border-radius: 18px;
            padding: 28px;
            text-align: center;
            background: rgba(15, 23, 42, 0.4);
        }

        .completed-clips-empty-title {
            color: #e2e8f0 !important;
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 8px;
        }

        .completed-clips-empty-copy {
            color: #94a3b8 !important;
            line-height: 1.5;
        }

        @media (max-width: 980px) {
            #hero-shell {
                width: calc(100vw - 28px);
            }

            #hero-shell {
                grid-template-columns: 1fr;
                justify-items: center;
            }

            #hero-title {
                grid-column: auto;
                font-size: 1rem;
                text-align: center;
            }

            #stage-card-header,
            #control-footer-row {
                flex-direction: column !important;
                align-items: center !important;
            }

            #control-actions-row {
                width: 100% !important;
                justify-content: center !important;
            }

            #upload-image-trigger,
            #upload-image-trigger .upload-image-trigger,
            #run-button {
                width: 100% !important;
            }

            .generation-badges {
                justify-content: flex-start;
            }

        }
        </style>
        """)
        gr.HTML("""
        <script>
        (() => {
            const marker = "data-fv-overscroll-init";
            const root = document.documentElement;
            if (root.getAttribute(marker) === "1") return;
            root.setAttribute(marker, "1");

            const focusPromptTextbox = () => {
                const textarea = document.querySelector("#prompt-textbox textarea");
                if (!textarea) return false;
                if (document.activeElement && document.activeElement !== document.body) return true;
                textarea.focus();
                textarea.setSelectionRange(0, 0);
                return true;
            };

            const focusPromptTextboxWithRetry = (attempts = 10) => {
                if (focusPromptTextbox() || attempts <= 0) return;
                window.setTimeout(() => focusPromptTextboxWithRetry(attempts - 1), 120);
            };

            let releaseTimer = null;

            const applyOverscrollShift = (value) => {
                root.style.setProperty("--fv-overscroll-shift", `${value}px`);
                if (releaseTimer) {
                    window.clearTimeout(releaseTimer);
                }
                releaseTimer = window.setTimeout(() => {
                    root.style.setProperty("--fv-overscroll-shift", "0px");
                }, 160);
            };

            window.addEventListener("wheel", (event) => {
                const scroller = document.scrollingElement || document.documentElement;
                const maxScroll = scroller.scrollHeight - window.innerHeight;
                const atTop = scroller.scrollTop <= 0 && event.deltaY < 0;
                const atBottom = scroller.scrollTop >= maxScroll - 1 && event.deltaY > 0;

                if (!atTop && !atBottom) {
                    return;
                }

                const shift = Math.max(-22, Math.min(22, -event.deltaY * 0.12));
                applyOverscrollShift(shift);
            }, { passive: true });

            window.setTimeout(() => focusPromptTextboxWithRetry(), 120);
        })();
        </script>
        """)
        
        def on_example_select(example_label):
            if example_label and example_label in example_labels:
                index = example_labels.index(example_label)
                return examples[index]
            return gr.update()
        
        example_dropdown.change(
            fn=on_example_select,
            inputs=example_dropdown,
            outputs=prompt_textbox,
        )

        def on_input_image_change(input_image):
            has_image = bool(input_image)
            return (
                render_input_image_status(input_image),
                gr.update(visible=has_image),
            )

        input_image.change(
            fn=on_input_image_change,
            inputs=input_image,
            outputs=[image_upload_status, clear_image_button],
        )

        def clear_selected_image():
            return (
                gr.update(value=None),
                render_input_image_status(None),
                gr.update(visible=False),
            )

        clear_image_button.click(
            fn=clear_selected_image,
            inputs=None,
            outputs=[input_image, image_upload_status, clear_image_button],
        )

        def on_model_selection_change(selected_model):
            height, width, num_frames = get_default_values(selected_model)
            return (
                gr.update(value=height),
                gr.update(value=width),
                gr.update(value=num_frames),
                render_generation_badges(selected_model),
            )

        model_selection.change(
            fn=on_model_selection_change,
            inputs=model_selection,
            outputs=[
                height_display,
                width_display,
                num_frames_display,
                stage_badges,
            ],
        )
        
        def summarize_clip_status(session_clips):
            session_clips = session_clips or []
            count = len(session_clips)
            if count == 0:
                status = "Your creations for this browser session will appear here."
            elif count == 1:
                status = "1 creation saved for this browser session."
            else:
                status = f"{count} creations saved for this browser session."
            return status

        def load_session_gallery(session_clips=None):
            session_clips = session_clips or []
            return (
                render_completed_clips(session_clips),
                summarize_clip_status(session_clips),
            )

        def clear_session_gallery():
            return (
                render_completed_clips([]),
                "Your creations for this browser session were cleared.",
                [],
            )

        def handle_generation(
            model_selection,
            prompt,
            input_image,
            session_clips=None,
        ):
            session_clips = session_clips or []
            normalized_prompt = prompt.strip()
            if not normalized_prompt:
                message = "Prompt is empty."
                gr.Warning(message)
                return (
                    gr.update(value=None, visible=True),
                    gr.update(
                        visible=True,
                        value=render_error_message(message),
                    ),
                    gr.update(visible=False, value=create_timing_placeholder()),
                    gr.update(visible=False),
                    render_completed_clips(session_clips),
                    summarize_clip_status(session_clips),
                    session_clips,
                )

            safety_check = get_prompt_safety_check(normalized_prompt)
            if safety_check.blocked:
                message = safety_check.message or "Prompt was blocked."
                gr.Warning(message)
                return (
                    gr.update(value=None, visible=True),
                    gr.update(
                        visible=True,
                        value=render_prompt_blocked_message(
                            message,
                            safety_check.category,
                        ),
                    ),
                    gr.update(visible=False, value=create_timing_placeholder()),
                    gr.update(visible=False),
                    render_completed_clips(session_clips),
                    summarize_clip_status(session_clips),
                    session_clips,
                )

            try:
                prompt_for_generation = maybe_enhance_prompt(
                    normalized_prompt,
                    curated_prompts,
                )
            except RuntimeError as error:
                message = str(error)
                gr.Warning(message)
                return (
                    gr.update(value=None, visible=True),
                    gr.update(
                        visible=True,
                        value=render_error_message(message),
                    ),
                    gr.update(visible=False, value=create_timing_placeholder()),
                    gr.update(visible=False),
                    render_completed_clips(session_clips),
                    summarize_clip_status(session_clips),
                    session_clips,
                )

            if prompt_for_generation != normalized_prompt:
                enhanced_safety_check = get_prompt_safety_check(
                    prompt_for_generation
                )
                if enhanced_safety_check.blocked:
                    message = (
                        "Prompt enhancement produced text that was blocked by "
                        "the safety filter. Please revise the prompt and try "
                        "again."
                    )
                    gr.Warning(message)
                    return (
                        gr.update(value=None, visible=True),
                        gr.update(
                            visible=True,
                            value=render_prompt_blocked_message(
                                message,
                                enhanced_safety_check.category,
                            ),
                        ),
                        gr.update(
                            visible=False,
                            value=create_timing_placeholder(),
                        ),
                        gr.update(visible=False),
                        render_completed_clips(session_clips),
                        summarize_clip_status(session_clips),
                        session_clips,
                    )

            result_path, seed_or_error, num_frames, generation_time, e2e_latency = generate_video(
                prompt_for_generation, model_selection, input_image
            )
            timing_details = create_timing_display(
                inference_time=generation_time,
                total_time=e2e_latency,
                stage_execution_times=[],
                num_frames=num_frames,
            )
            if result_path and os.path.exists(result_path):
                session_clips = _record_session_clip(
                    session_clips,
                    output_path=result_path,
                    prompt=prompt_for_generation,
                    model_name=model_selection,
                    num_frames=num_frames,
                    generation_time=generation_time,
                )
                return (
                    gr.update(value=result_path, visible=True),
                    gr.update(visible=False),
                    gr.update(visible=True, value=timing_details),
                    gr.update(visible=True),
                    render_completed_clips(session_clips),
                    summarize_clip_status(session_clips),
                    session_clips,
                )
            else:
                return (
                    gr.update(value=None, visible=True),
                    gr.update(
                        visible=True,
                        value=render_error_message(str(seed_or_error)),
                    ),
                    gr.update(visible=False, value=create_timing_placeholder()),
                    gr.update(visible=False),
                    render_completed_clips(session_clips),
                    summarize_clip_status(session_clips),
                    session_clips,
                )

        demo.load(
            fn=load_session_gallery,
            outputs=[completed_clips_html, completed_clips_status],
        )

        clear_clips_button.click(
            fn=clear_session_gallery,
            outputs=[
                completed_clips_html,
                completed_clips_status,
                completed_clips_state,
            ],
            queue=False,
        )
        
        run_button.click(
            fn=handle_generation,
            inputs=[
                model_selection,
                prompt_textbox,
                input_image,
                completed_clips_state,
            ],
            outputs=[
                result,
                error_output,
                timing_display,
                timing_title,
                completed_clips_html,
                completed_clips_status,
                completed_clips_state,
            ],
            concurrency_limit=1,
            show_progress_on=result,
            queue=False,
        )
    
    return demo
