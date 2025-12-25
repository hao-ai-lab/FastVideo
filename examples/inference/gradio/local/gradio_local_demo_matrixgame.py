import argparse
import asyncio
import os
import time

import gradio as gr
import imageio
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse

from fastvideo.entrypoints.streaming_generator import StreamingVideoGenerator
from fastvideo.models.dits.matrix_game.utils import expand_action_to_frames


VARIANT_CONFIG = {
    "Matrix-Game-2.0-Base": {
        "model_path": "FastVideo/Matrix-Game-2.0-Base-Diffusers",
        "keyboard_dim": 4,
        "mode": "universal",
        "image_url": "https://raw.githubusercontent.com/SkyworkAI/Matrix-Game/main/Matrix-Game-2/demo_images/universal/0000.png",
    },
    "Matrix-Game-2.0-GTA": {
        "model_path": "FastVideo/Matrix-Game-2.0-GTA-Diffusers",
        "keyboard_dim": 2,
        "mode": "gta_drive",
        "image_url": "https://raw.githubusercontent.com/SkyworkAI/Matrix-Game/main/Matrix-Game-2/demo_images/gta_drive/0000.png",
    },
    "Matrix-Game-2.0-TempleRun": {
        "model_path": "FastVideo/Matrix-Game-2.0-TempleRun-Diffusers",
        "keyboard_dim": 7,
        "mode": "templerun",
        "image_url": "https://raw.githubusercontent.com/SkyworkAI/Matrix-Game/main/Matrix-Game-2/demo_images/temple_run/0000.png",
    },
}

MODEL_PATH_MAPPING = {
    name: config["model_path"] for name, config in VARIANT_CONFIG.items()
}


CAM_VALUE = 0.1
KEYBOARD_MAP_UNIVERSAL = {
    "W (Forward)": [1, 0, 0, 0],
    "S (Back)": [0, 1, 0, 0],
    "A (Left)": [0, 0, 1, 0],
    "D (Right)": [0, 0, 0, 1],
    "Q (Stop)": [0, 0, 0, 0],
}
KEYBOARD_MAP_GTA = {
    "W (Forward)": [1, 0],
    "S (Back)": [0, 1],
    "Q (Stop)": [0, 0],
}
KEYBOARD_MAP_TEMPLERUN = {
    "Q (Run)": [1, 0, 0, 0, 0, 0, 0],
    "W (Jump)": [0, 1, 0, 0, 0, 0, 0],
    "S (Slide)": [0, 0, 1, 0, 0, 0, 0],
    "Z (Turn Left)": [0, 0, 0, 1, 0, 0, 0],
    "C (Turn Right)": [0, 0, 0, 0, 1, 0, 0],
    "A (Left)": [0, 0, 0, 0, 0, 1, 0],
    "D (Right)": [0, 0, 0, 0, 0, 0, 1],
}


CAMERA_MAP_UNIVERSAL = {
    "U (Center)": [0, 0],
    "I (Up)": [CAM_VALUE, 0],
    "K (Down)": [-CAM_VALUE, 0],
    "J (Left)": [0, -CAM_VALUE],
    "L (Right)": [0, CAM_VALUE],
}
CAMERA_MAP_GTA = {
    "A (Steer Left)": [0, -CAM_VALUE],
    "D (Steer Right)": [0, CAM_VALUE],
    "Q (Straight)": [0, 0],
}

def setup_model_environment(model_path: str) -> None:
    # if "fullattn" in model_path.lower():
    #     os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "FLASH_ATTN"
    # else:
    #     os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "VIDEO_SPARSE_ATTN"
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "FLASH_ATTN"
    os.environ["FASTVIDEO_STAGE_LOGGING"] = "1"

def create_timing_display(inference_time, total_time, stage_execution_times, num_frames):
    dit_denoising_time = f"{stage_execution_times[5]:.2f}s" if len(stage_execution_times) > 5 else "N/A"
    
    timing_html = f"""
    <div style="margin: 10px 0;">
        <h3 style="text-align: center; margin-bottom: 10px;">⏱️ Timing Breakdown</h3>
        <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-bottom: 10px;">
            <div class="timing-card timing-card-highlight">
                <div style="font-size: 20px;">🚀</div>
                <div style="font-weight: bold; margin: 3px 0; font-size: 14px;">DiT Denoising</div>
                <div style="font-size: 16px; color: #ffa200; font-weight: bold;">{dit_denoising_time}</div>
            </div>
            <div class="timing-card">
                <div style="font-size: 20px;">🧠</div>
                <div style="font-weight: bold; margin: 3px 0; font-size: 14px;">E2E (w. vae/text encoder)</div>
                <div style="font-size: 16px; color: #2563eb;">{inference_time:.2f}s</div>
            </div>
            <div class="timing-card">
                <div style="font-size: 20px;">🎬</div>
                <div style="font-weight: bold; margin: 3px 0; font-size: 14px;">Video Encoding</div>
                <div style="font-size: 16px; color: #dc2626;">N/A</div>
            </div>
            <div class="timing-card">
                <div style="font-size: 20px;">🌐</div>
                <div style="font-weight: bold; margin: 3px 0; font-size: 14px;">Network Transfer</div>
                <div style="font-size: 16px; color: #059669;">N/A</div>
            </div>
            <div class="timing-card">
                <div style="font-size: 20px;">📊</div>
                <div style="font-weight: bold; margin: 3px 0; font-size: 14px;">Total Processing</div>
                <div style="font-size: 18px; color: #0277bd;">{total_time:.2f}s</div>
            </div>
        </div>"""
    
    if inference_time > 0:
        fps = num_frames / inference_time
        timing_html += f"""
        <div class="performance-card" style="margin-top: 15px;">
            <span style="font-weight: bold;">Generation Speed: </span>
            <span style="font-size: 18px; color: #6366f1; font-weight: bold;">{fps:.1f} frames/second</span>
        </div>"""
    
    return timing_html + "</div>"

def get_action_tensors(mode: str, keyboard_key: str, mouse_key: str | None):
    if mode == "universal":
        keyboard = torch.tensor(KEYBOARD_MAP_UNIVERSAL.get(keyboard_key, [0, 0, 0, 0])).cuda()
        mouse = torch.tensor(CAMERA_MAP_UNIVERSAL.get(mouse_key, [0, 0])).cuda()
    elif mode == "gta_drive":
        keyboard = torch.tensor(KEYBOARD_MAP_GTA.get(keyboard_key, [0, 0])).cuda()
        mouse = torch.tensor(CAMERA_MAP_GTA.get(mouse_key, [0, 0])).cuda()
    elif mode == "templerun":
        keyboard = torch.tensor(KEYBOARD_MAP_TEMPLERUN.get(keyboard_key, [1, 0, 0, 0, 0, 0, 0])).cuda()
        mouse = None
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return {"keyboard": keyboard, "mouse": mouse}

def create_gradio_interface(generators: dict[str, StreamingVideoGenerator], loaded_model_name: str):
    initial_config = VARIANT_CONFIG.get(loaded_model_name, VARIANT_CONFIG["Matrix-Game-2.0-Base"])
    initial_mode = initial_config["mode"]
    
    if initial_mode == "universal":
        initial_kb_choices = list(KEYBOARD_MAP_UNIVERSAL.keys())
        initial_mouse_choices = list(CAMERA_MAP_UNIVERSAL.keys())
        initial_mouse_visible = True
    elif initial_mode == "gta_drive":
        initial_kb_choices = list(KEYBOARD_MAP_GTA.keys())
        initial_mouse_choices = list(CAMERA_MAP_GTA.keys())
        initial_mouse_visible = True
    else:  # templerun
        initial_kb_choices = list(KEYBOARD_MAP_TEMPLERUN.keys())
        initial_mouse_choices = []
        initial_mouse_visible = False
    
    theme = gr.themes.Base().set(
        button_primary_background_fill="#2563eb",
        button_primary_background_fill_hover="#1d4ed8",
        button_primary_text_color="white",
        slider_color="#2563eb",
        checkbox_background_color_selected="#2563eb",
    )
    
    with gr.Blocks(title="FastVideo - Matrix Game 2.0", theme=theme) as demo:
        game_state = gr.State({
            "initialized": False,
            "current_model": None,
            "block_idx": 0,
            "max_blocks": 50,
        })
        
        # Header
        gr.Image("assets/full.svg", show_label=False, container=False, height=80)
        
        gr.HTML("""
        <div style="text-align: center; margin-bottom: 10px;">
            <p style="font-size: 18px;"> Make Video Generation Go Blurrrrrrr </p>
            <p style="font-size: 18px;"> <a href="https://github.com/hao-ai-lab/FastVideo/tree/main" target="_blank">Code</a> | <a href="https://hao-ai-lab.github.io/blogs/fastvideo_post_training/" target="_blank">Blog</a> | <a href="https://hao-ai-lab.github.io/FastVideo/" target="_blank">Docs</a>  </p>
        </div>
        """)
        
        with gr.Accordion("🎥 What Is FastVideo?", open=False):
            gr.HTML("""
            <div style="padding: 20px; line-height: 1.6;">
                <p style="font-size: 16px; margin-bottom: 15px;">
                    FastVideo is an inference and post-training framework for diffusion models. It features an end-to-end unified pipeline for accelerating diffusion models, starting from data preprocessing to model training, finetuning, distillation, and inference. FastVideo is designed to be modular and extensible, allowing users to easily add new optimizations and techniques. Whether it is training-free optimizations or post-training optimizations, FastVideo has you covered.
                </p>
            </div>
            """)
        
        # Model Selection
        with gr.Row():
            model_selection = gr.Dropdown(
                choices=[loaded_model_name],
                value=loaded_model_name,
                label="Select Model",
                interactive=False
            )
        

        # Main Layout
        with gr.Row(equal_height=True, elem_classes="main-content-row"):
            with gr.Column(scale=1, elem_classes="advanced-options-column"):
                with gr.Group():
                    gr.HTML("<div style='margin: 0 0 15px 0; text-align: center; font-size: 16px;'>Game Controls</div>")
                    
                    with gr.Group():
                        gr.HTML("<div style='font-size: 14px; margin-bottom: 5px; font-weight: bold;'>🎮 Keyboard Control</div>")
                        keyboard_action = gr.Radio(
                            choices=initial_kb_choices,
                            value=initial_kb_choices[0] if initial_kb_choices else None,
                            label="Movement",
                            show_label=False,
                            interactive=True
                        )
                    
                    with gr.Group(visible=initial_mouse_visible) as mouse_group:
                        gr.HTML("<div style='font-size: 14px; margin-bottom: 5px; font-weight: bold;'>🖱️ Mouse/Camera Control</div>")
                        mouse_action = gr.Radio(
                            choices=initial_mouse_choices if initial_mouse_visible else [],
                            value=initial_mouse_choices[0] if initial_mouse_choices else None,
                            label="Camera",
                            show_label=False,
                            interactive=True
                        )
                    
                    with gr.Row():
                        action_btn = gr.Button("Start", variant="primary")
                        stop_btn = gr.Button("Stop", variant="stop")
                    
                    gr.HTML("<div style='margin-top: 15px;'></div>")
                    
                    seed = gr.Slider(
                        label="Seed",
                        minimum=0,
                        maximum=1000000,
                        step=1,
                        value=1024,
                    )
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=False)
                    seed_output = gr.Number(label="Used Seed")
                    
                    block_counter = gr.Textbox(label="Progress", value="Block: 0 / 50", interactive=False, lines=1)


            # Right Column: Video Output
            with gr.Column(scale=1, elem_classes="video-column"):
                video_output = gr.Video(
                    label="Generated Video",
                    show_label=True,
                    height=466,
                    width=600,
                    container=True,
                    elem_classes="video-component",
                    autoplay=True
                )

        # Styles
        gr.HTML("""
        <style>
        .center-button {
            display: flex !important;
            justify-content: center !important;
            height: 100% !important;
            padding-top: 1.4em !important;
        }
        
        .gradio-container {
            max-width: 1200px !important;
            margin: 0 auto !important;
        }
        
        .main {
            max-width: 1200px !important;
            margin: 0 auto !important;
        }
        
        .gr-form, .gr-box, .gr-group {
            max-width: 1200px !important;
        }
        
        .gr-video {
            max-width: 500px !important;
            margin: 0 auto !important;
        }
        
        .main-content-row {
            display: flex !important;
            align-items: flex-start !important;
            min-height: 500px !important;
            gap: 20px !important;
        }
        
        .advanced-options-column,
        .video-column {
            display: flex !important;
            flex-direction: column !important;
            flex: 1 !important;
            min-height: 400px !important;
            align-items: stretch !important;
        }
        
        .video-column > * {
            margin-top: 0 !important;
        }
        
        .video-column .gr-video,
        .video-component {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
        
        .video-column .gr-video .gr-form {
            margin-top: 0 !important;
        }
        
        .advanced-options-column .gr-group,
        .video-column .gr-video {
            margin-top: 0 !important;
            vertical-align: top !important;
        }
        
        .advanced-options-column > *:last-child,
        .video-column > *:last-child {
            flex-grow: 0 !important;
        }
        
        @media (max-width: 1400px) {
            .main-content-row {
                min-height: 600px !important;
            }
            
            .advanced-options-column,
            .video-column {
                min-height: 600px !important;
            }
        }
        
        @media (max-width: 1200px) {
            .main-content-row {
                flex-direction: column !important;
                align-items: stretch !important;
            }
            
            .advanced-options-column,
            .video-column {
                min-height: auto !important;
                width: 100% !important;
            }
        }
        
        .timing-card {
            background: var(--background-fill-secondary) !important;
            border: 1px solid var(--border-color-primary) !important;
            color: var(--body-text-color) !important;
            padding: 10px;
            border-radius: 8px;
            text-align: center;
            min-height: 80px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        
        .timing-card-highlight {
            background: var(--background-fill-primary) !important;
            border: 2px solid var(--color-accent) !important;
        }
        
        .performance-card {
            background: var(--background-fill-secondary) !important;
            border: 1px solid var(--border-color-primary) !important;
            color: var(--body-text-color) !important;
            padding: 10px;
            border-radius: 6px;
            text-align: center;
        }

        .gr-number input[readonly] {
            background-color: var(--background-fill-secondary) !important;
            border: 1px solid var(--border-color-primary) !important;
            color: var(--body-text-color-subdued) !important;
            cursor: default !important;
            text-align: center !important;
            font-weight: 500 !important;
        }
        </style>
        """)
        
        # UI update based on model selection
        def on_model_change(model_name):
            config = VARIANT_CONFIG.get(model_name, VARIANT_CONFIG["Matrix-Game-2.0-Base"])
            mode = config["mode"]
            
            if mode == "universal":
                kb_choices = list(KEYBOARD_MAP_UNIVERSAL.keys())
                mouse_choices = list(CAMERA_MAP_UNIVERSAL.keys())
                mouse_visible = True
            elif mode == "gta_drive":
                kb_choices = list(KEYBOARD_MAP_GTA.keys())
                mouse_choices = list(CAMERA_MAP_GTA.keys())
                mouse_visible = True
            else:  # templerun
                kb_choices = list(KEYBOARD_MAP_TEMPLERUN.keys())
                mouse_choices = []
                mouse_visible = False
            
            return (
                gr.update(choices=kb_choices, value=kb_choices[0] if kb_choices else None),
                gr.update(choices=mouse_choices, value=mouse_choices[0] if mouse_choices else None, visible=mouse_visible),
                gr.update(visible=mouse_visible),
            )
        
        model_selection.change(
            fn=on_model_change,
            inputs=model_selection,
            outputs=[keyboard_action, mouse_action, mouse_group]
        )

        def start_game(model_name, seed_val, randomize, state):
            if randomize:
                seed_val = torch.randint(0, 1000000, (1,)).item()
            
            config = VARIANT_CONFIG.get(model_name)
            if not config:
                return state, seed_val, "Block: 0 / 50", None, "", gr.update(), gr.update()
            
            generator = generators.get(config["model_path"])
            if not generator:
                return state, seed_val, "Block: 0 / 50", None, "", gr.update(), gr.update()
            
            # If already initialized, clean up first
            if state.get("initialized"):
                try:
                    # Clear accumulated frames without saving
                    generator.accumulated_frames = []
                    generator.executor.execute_streaming_clear()
                except Exception as e:
                    print(f"Warning: cleanup error: {e}")
            
            # Streaming parameters
            num_latent_frames_per_block = 3
            max_blocks = 50
            total_latent_frames = num_latent_frames_per_block * max_blocks
            num_frames = (total_latent_frames - 1) * 4 + 1
            
            actions = {
                "keyboard": torch.zeros((num_frames, config["keyboard_dim"])),
                "mouse": torch.zeros((num_frames, 2))
            }
            grid_sizes = torch.tensor([150, 44, 80])
            
            output_dir = os.path.abspath("outputs/matrixgame")
            os.makedirs(output_dir, exist_ok=True)
            video_path = os.path.join(output_dir, f"video_{int(time.time())}.mp4")
            
            generator.reset(
                prompt="",
                image_path=config["image_url"],
                mouse_cond=actions["mouse"].unsqueeze(0),
                keyboard_cond=actions["keyboard"].unsqueeze(0),
                grid_sizes=grid_sizes,
                num_frames=num_frames,
                height=352,
                width=640,
                num_inference_steps=50,
                output_path=video_path,
            )
            
            new_state = {
                "initialized": True,
                "current_model": model_name,
                "block_idx": 0,
                "max_blocks": max_blocks,
                "video_path": video_path,
                "frames_per_block": num_latent_frames_per_block * 4,
                "mode": config["mode"],
                "seed": seed_val,
            }
            
            return new_state, seed_val, "Block: 0 / 50", None, gr.update(value="Step"), gr.update(interactive=True)
        
        async def step_game(keyboard_key, mouse_key, model_name, state):
            if not state.get("initialized"):
                return state, state.get("seed", 0), "Block: 0 / 50", None, gr.update(), gr.update()
            
            total_start_time = time.time()
            config = VARIANT_CONFIG.get(model_name)
            generator = generators.get(config["model_path"])
            mode = state["mode"]
            frames_per_block = state["frames_per_block"]

            # Parse inputs to tensors
            action = get_action_tensors(mode, keyboard_key, mouse_key)
            keyboard_cond, mouse_cond = expand_action_to_frames(action, frames_per_block)
            
            # run step async
            inference_start_time = time.time()
            frames, block_future = await generator.step_async(keyboard_cond, mouse_cond)
            inference_time = time.time() - inference_start_time
            
            # wait for block file to be written
            block_path = block_future.result() if block_future else None
            state["block_idx"] = generator.block_idx
            block_str = f"Block: {state['block_idx']} / {state['max_blocks']}"
            
            total_time = time.time() - total_start_time
            
            # Timing breakdown
            timing_html = create_timing_display(inference_time, total_time, [], frames_per_block)

            return state, state.get("seed", 0), block_str, block_path, gr.update(), gr.update()

        def stop_game(model_name, state):
            if not state.get("initialized"):
                return {"initialized": False}, 0, "Block: 0 / 50", None, gr.update(value="Start"), gr.update(interactive=False)
            
            config = VARIANT_CONFIG.get(model_name)
            generator = generators.get(config["model_path"])
            
            final_path = state.get("video_path")
            generator.finalize(final_path)
            
            return {"initialized": False}, state.get("seed", 0), "Block: 0 / 50", final_path, gr.update(value="Start"), gr.update(interactive=False)
        
        async def handle_action(keyboard_key, mouse_key, model_name, seed_val, randomize, state):
            if not state.get("initialized"):
                return start_game(model_name, seed_val, randomize, state)
            else:
                return await step_game(keyboard_key, mouse_key, model_name, state)
        
        action_btn.click(
            fn=handle_action,
            inputs=[keyboard_action, mouse_action, model_selection, seed, randomize_seed, game_state],
            outputs=[game_state, seed_output, block_counter, video_output, action_btn, stop_btn]
        )
        
        stop_btn.click(
            fn=stop_game,
            inputs=[model_selection, game_state],
            outputs=[game_state, seed_output, block_counter, video_output, action_btn, stop_btn]
        )
        
        gr.HTML("""
        <div style="text-align: center; margin-top: 10px; margin-bottom: 15px;">
            <p style="font-size: 16px; margin: 0;">Note that this demo is meant to showcase Matrix Game's quality and that under a large number of requests, generation speed may be affected.</p>
        </div>
        """)
    
    return demo


def main():
    parser = argparse.ArgumentParser(description="Matrix Game Gradio Demo")
    parser.add_argument("--model", type=str, default="Matrix-Game-2.0-Base",
                        choices=list(VARIANT_CONFIG.keys()),
                        help="Model variant to load")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    
    # Load the selected model
    config = VARIANT_CONFIG[args.model]
    model_path = config["model_path"]
    
    print(f"Loading model: {model_path}")
    setup_model_environment(model_path)
    generator = StreamingVideoGenerator.from_pretrained(
        model_path,
        num_gpus=1,
        use_fsdp_inference=True,
        dit_cpu_offload=True,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=True,
        pin_cpu_memory=True,
    )
    
    generators = {model_path: generator}
    
    demo = create_gradio_interface(generators, args.model)
    
    print(f"Starting Gradio at http://{args.host}:{args.port}")
    
    # FastAPI Wrapper
    app = FastAPI()
    
    @app.get("/logo.png")
    def get_logo():
        return FileResponse(
            "assets/full.svg",
            media_type="image/svg+xml",
            headers={
                "Cache-Control": "public, max-age=3600",
                "Access-Control-Allow-Origin": "*"
            }
        )

    @app.get("/favicon.ico")
    def get_favicon():
        favicon_path = "assets/icon-simple.svg"
        
        if os.path.exists(favicon_path):
            return FileResponse(
                favicon_path, 
                media_type="image/svg+xml",
                headers={
                    "Cache-Control": "public, max-age=3600",
                    "Access-Control-Allow-Origin": "*"
                }
            )
        else:
            raise HTTPException(status_code=404, detail="Favicon not found")
    
    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        base_url = str(request.base_url).rstrip('/')
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0" />
            
            <title>FastVideo - Matrix Game 2.0</title>
            <meta name="title" content="MatrixGame2.0">
            <meta name="description" content="Make video generation go blurrrrrrr">
            <meta name="keywords" content="FastVideo, video generation, AI, machine learning, Matrix Game 2.0">
            
            <meta property="og:type" content="website">
            <meta property="og:url" content="{base_url}/">
            <meta property="og:title" content="FastVideo - Matrix Game 2.0">
            <meta property="og:description" content="Make video generation go blurrrrrrr">
            <meta property="og:image" content="{base_url}/logo.png">
            <meta property="og:image:width" content="1200">
            <meta property="og:image:height" content="630">
            <meta property="og:site_name" content="MatrixGame2.0">
            
            <meta property="twitter:card" content="summary_large_image">
            <meta property="twitter:url" content="{base_url}/">
            <meta property="twitter:title" content="MatrixGame2.0">
            <meta property="twitter:description" content="Make video generation go blurrrrrrr">
            <meta property="twitter:image" content="{base_url}/logo.png">
            <link rel="icon" type="image/png" sizes="32x32" href="/favicon.ico">
            <link rel="icon" type="image/png" sizes="16x16" href="/favicon.ico">
            <link rel="apple-touch-icon" href="/favicon.ico">
            <style>
                body, html {{
                    margin: 0;
                    padding: 0;
                    height: 100%;
                    overflow: hidden;
                }}
                iframe {{
                    width: 100%;
                    height: 100vh;
                    border: none;
                }}
            </style>
        </head>
        <body>
            <iframe src="/gradio" width="100%" height="100%" style="border: none;"></iframe>
        </body>
        </html>
        """

    app = gr.mount_gradio_app(
        app, 
        demo, 
        path="/gradio",
        allowed_paths=[os.path.abspath("outputs"), os.path.abspath("fastvideo-logos")]
    )
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
