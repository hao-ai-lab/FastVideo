import os
import gradio as gr
import requests
from loguru import logger
import time
import base64
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
from pathlib import Path

API_PORT = int(os.getenv("API_PORT", 8082))
GRADIO_PORT = int(os.getenv("GRADIO_PORT", 8080))
VIDEO_ENC = os.getenv("VIDEO_ENC", "vp09")
BACKEND_URL = f"http://localhost:{API_PORT}/generate_next"
HEALTH_URL = f"http://localhost:{API_PORT}/health"
OUTPUT_DIR = "./gradio_results/"

PRESET_EXAMPLES = [
    {
        "image_path": "asset/village.png",
        "prompt": "A charming medieval village with cobblestone streets, thatched-roof houses, and vibrant flower gardens under a bright blue sky.",
        "name": "Medieval Village"
    },
    {
        "image_path": "asset/temple.png",
        "prompt": "A majestic ancient temple stands under a clear blue sky, its grandeur highlighted by towering Doric columns and intricate architectural details.",
        "name": "Ancient Temple"
    }
]

ACTION_MAP = {
    "forward": 'w',
    "left": 'a',
    "right": 'd',
    "backward": 's',
    "up_rot": "up_rot",
    "right_rot": "right_rot",
    "left_rot": "left_rot",
    "down_rot": "down_rot"
}

previous_video_path = None
previous_first_frame = None
all_video_paths = []
current_input_type = None  # record input type: "image" or "video"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_preset_by_selection(evt: gr.SelectData):
    selected_index = evt.index 
    # print(f"select index: {selected_index}")
    if selected_index is None:
        return None, ""
    if 0 <= selected_index < len(PRESET_EXAMPLES):
        example = PRESET_EXAMPLES[selected_index]
        # img = Image.open(example["image_path"]) if os.path.exists(example["image_path"]) else None
        return example["image_path"], example["prompt"]
    return None, ""

def image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def get_first_frame(video_path):
    if not video_path or not os.path.exists(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    cap.release()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)
    return None

def check_video_validity(video_path):
    """at least 33 frames"""
    if not video_path or not os.path.exists(video_path):
        return False, "Video file not found"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, "Could not open video file"
    frame_count = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        frame_count += 1
    cap.release()
    if frame_count == 0:
        return False, "Video file contains no frames"
    elif frame_count < 33:
        return False, f"Video file has insufficient frames ({frame_count}/33 required)"
    
    return True, f"Valid video file (contains {frame_count} frames)"

def concatenate_videos(video_paths, output_path):
    if not video_paths or len(video_paths) < 2:
        return video_paths[0] if video_paths else None
    first_cap = cv2.VideoCapture(video_paths[0])
    if not first_cap.isOpened():
        return None
    frame_width = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = first_cap.get(cv2.CAP_PROP_FPS) or 30.0
    first_cap.release()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*VIDEO_ENC)
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    if not out.isOpened():
        return None
    for path in video_paths:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            continue
        current_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        current_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame.shape[1] != frame_width or frame.shape[0] != frame_height:
                frame = cv2.resize(frame, (frame_width, frame_height))
            out.write(frame)
        cap.release()
    out.release()
    return output_path

def check_backend_healthy():
    try:
        logger.debug(f"Attempting health check request to {HEALTH_URL}")
        response = requests.get(
            HEALTH_URL,
            timeout=5,
            proxies={'http': '', 'https': ''},
            allow_redirects=False,
            headers={'Connection': 'close'} 
        )
        logger.debug(f"Health check response status: {response.status_code}")
        if response.status_code == 200:
            try:
                data = response.json()
                logger.debug(f"Health check response data: {data}")
                is_healthy = data.get("status") == "healthy"
                logger.debug(f"API reports healthy: {is_healthy}")
                return is_healthy
            except ValueError as json_err:
                logger.error(f"Failed to parse JSON response: {json_err}")
                return False
        else:
            logger.warning(f"Health check returned non-200 status: {response.status_code}")
            logger.debug(f"Response text (first 200 chars): {response.text[:200]}")
    except requests.exceptions.Timeout:
        logger.debug(f"Health check timed out.")
    except requests.exceptions.ConnectionError as conn_err:
        logger.debug(f"Health check connection error: {conn_err}")
        if ' refused' in str(conn_err).lower() or 'name resolution' in str(conn_err).lower():
             logger.debug("Connection error might be related to localhost resolution or service not running.")
    except requests.exceptions.RequestException as e:
        logger.debug(f"Health check failed with RequestException: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during health check: {e}")
    return False

def generate_next_video():
    try:
        logger.info("Sending request to API server...")
        response = requests.post(
            BACKEND_URL,
            timeout=6000,
            proxies={'http': '', 'https': ''},
            headers={'Connection': 'close'},
            allow_redirects=False,
            json={}
        )
        logger.info(f"Received response from API server: {response.status_code}")
        if response.status_code != 200:
             try:
                 error_data = response.json()
                 return None, f"❌ API Error ({response.status_code}): {error_data.get('error', 'Unknown error')}"
             except:
                 return None, f"❌ API Error ({response.status_code}): {response.text}"
        result = response.json()
        if "error" in result:
            return None, f"❌ {result['error']}"
        video_path = result.get("video_path")
        if video_path and os.path.exists(video_path):
            return video_path, f"✅ Generated index {result['index']}"
        else:
            return None, "❌ Video file not found or path invalid."
    except requests.exceptions.Timeout:
        return None, "❌ Request timeout. Video generation might still be running."
    except requests.exceptions.ConnectionError as conn_err:
         logger.error(f"Connection error when sending request to API: {conn_err}")
         return None, f"❌ Connection failed: Could not reach API server. Check if it's running and not blocked by proxy. Details: {conn_err}"
    except requests.exceptions.RequestException as e:
        return None, f"❌ Request failed: {str(e)}"
    except Exception as e:
        return None, f"❌ Unexpected error: {str(e)}"

def generate_custom_video(
    input_content,
    input_type,
    video_continue,
    selected_actions,
    action_speed_list_str,
    input_prompt,
    add_pos_prompt,
    add_neg_prompt,
    video_size_width,
    video_size_height,
    cfg_scale,
    seed,
    sample_n_frames,
    infer_steps,
    flow_shift_eval_video
):
    global previous_video_path, previous_first_frame, all_video_paths, current_input_type
    try:
        num_actions = len(selected_actions)
        try:
            action_speed_list_temp = list(map(float, action_speed_list_str.split()))
            num_speeds = len(action_speed_list_temp)
        except ValueError:
            return None, [], "Invalid format for Action Speed List. Please enter numbers separated by spaces.", gr.update(interactive=False)
        if num_actions != num_speeds:
            return None, [], f"The number of actions ({num_actions}) does not match the number of speeds ({num_speeds}). Please ensure both lists have the same length.", gr.update(interactive=False)
        try:
            action_list = [ACTION_MAP[action] for action in selected_actions]
        except KeyError as e:
            return None, [], f"Invalid action name: {e}. Please select from the dropdown list.", gr.update(interactive=False)
        logger.info(f"Converted actions {selected_actions} to action_list (IDs): {action_list}")
        action_speed_list = list(map(float, action_speed_list_str.split()))
        base64_img = ""
        video_path = None
        image_start = True

        if input_type == "video":
            # check_video_validity
            is_valid, msg = check_video_validity(input_content)
            if not is_valid:
                return None, [], f"Invalid video: {msg}", gr.update(interactive=False)
            
            first_frame = get_first_frame(input_content)
            if not first_frame:
                return None, [], "Could not extract first frame from video", gr.update(interactive=False)
            
            base64_img = image_to_base64(first_frame)
            video_path = input_content 
            image_start = False
            previous_video_path = video_path

        
        if previous_video_path is not None and video_continue:
            # after image_start to continuation
            image_start = False
            video_path = previous_video_path
            previous_first_frame = get_first_frame(video_path)
            base64_img = image_to_base64(previous_first_frame)
        else:
            img = Image.open(input_content)
            base64_img = image_to_base64(img)
            
            previous_video_path = None
            previous_first_frame = None
            all_video_paths = []
            image_start = True

        payload = {
            "custom_params": {
                "input_image": base64_img,
                "input_prompt": input_prompt,
                "add_pos_prompt": add_pos_prompt,
                "add_neg_prompt": add_neg_prompt,
                "video_size": [int(video_size_width), int(video_size_height)],
                "cfg_scale": float(cfg_scale),
                "image_start": image_start,
                "action_list": action_list,
                "action_speed_list": action_speed_list,
                "seed": int(seed),
                "sample_n_frames": int(sample_n_frames),
                "infer_steps": int(infer_steps),
                "flow_shift_eval_video": float(flow_shift_eval_video),
                "save_path": OUTPUT_DIR
            }
        }
        if video_path:
            payload["custom_params"]["video_path"] = video_path
        
        payload_without_image = payload.copy()
        if "custom_params" in payload_without_image:
            custom_params = payload_without_image["custom_params"].copy()
            custom_params.pop("input_image", None)
            payload_without_image["custom_params"] = custom_params
        print(previous_video_path, video_continue, "\n", payload_without_image)
        logger.info("Sending custom request to API server...")
        response = requests.post(
            BACKEND_URL,
            timeout=36000,
            proxies={'http': '', 'https': ''},
            headers={'Connection': 'close'},
            allow_redirects=False,
            json=payload
        )
        logger.info(f"Received response from API server: {response.status_code}")
        if response.status_code != 200:
            try:
                error_data = response.json()
                return None, [], f"❌ API Error ({response.status_code}): {error_data.get('error', 'Unknown error')}", gr.update(interactive=bool(previous_video_path))
            except:
                return None, [], f"❌ API Error ({response.status_code}): {response.text}", gr.update(interactive=bool(previous_video_path))
        result = response.json()
        if "error" in result:
            return None, [], f"❌ {result['error']}", gr.update(interactive=bool(previous_video_path))
        video_path = result.get("video_path")
        frame_paths = result.get("frame_paths", [])
        if video_path and os.path.exists(video_path):
            previous_video_path = video_path
            previous_first_frame = get_first_frame(video_path)
            all_video_paths.append(video_path)
            concatenated_path = None
            if len(all_video_paths) > 1:
                concat_filename = f"concatenated_{int(time.time())}.mp4"
                concatenated_path = os.path.join(OUTPUT_DIR, concat_filename)
                concatenated_path = concatenate_videos(all_video_paths, concatenated_path)
            
            # update input type
            current_input_type = input_type
            if input_type == "video":
                return (concatenated_path if concatenated_path else video_path, 
                        frame_paths, 
                        f"✅ Generated index {result.get('index', 'N/A')} saving at {video_path}//{concatenated_path}",
                        gr.update(interactive=False, value=True))
            else:
                return (concatenated_path if concatenated_path else video_path, 
                        frame_paths, 
                        f"✅ Generated index {result.get('index', 'N/A')} saving at {video_path}//{concatenated_path}",
                        gr.update(interactive=True, value=True))
        else:
            return None, [], "❌ Video file not found or path invalid.", gr.update(interactive=bool(previous_video_path))
    except Exception as e:
        logger.exception("Error in generate_custom_video")
        return None, [], f"❌ Unexpected error: {str(e)}", gr.update(interactive=bool(previous_video_path))

def handle_input_change(input_content):
    global previous_video_path, previous_first_frame, all_video_paths, current_input_type
    
    if input_content is not None:
        # reset state
        previous_video_path = None
        previous_first_frame = None
        all_video_paths = []
        
        # detect input type
        if isinstance(input_content, str) and input_content.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
            current_input_type = "video"
            # Video Continuation=True(cannot change)
            return gr.update(value=True, interactive=False), "video"
        else:
            current_input_type = "image"
            return gr.update(value=False, interactive=False), "image"
    else:
        current_input_type = None
        return gr.update(value=False, interactive=False), None

def update_preview(file_info):
    if not file_info:
        return gr.update(visible=False), gr.update(visible=False)
    
    file_path = file_info.name
    file_name = file_path.lower()
    
    if file_name.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        return gr.update(value=file_path, visible=True), gr.update(visible=False)
    elif file_name.endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm')):
        return gr.update(visible=False), gr.update(value=file_path, visible=True)
    return gr.update(visible=False), gr.update(visible=False)

if __name__ == "__main__":
    logger.info("Checking if API server is ready...")
    max_wait = 60
    waited = 0
    while waited < max_wait:
        if check_backend_healthy():
            logger.info("✅ API server is ready.")
            break
        time.sleep(5)
        waited += 5
        logger.debug(f"Still waiting for API server... {waited}/{max_wait}s")
    else:
        logger.error("❌ API server not ready after timeout. Please check api_server.py logs.")
        print("Please ensure the API server (api_server.py) is running and healthy.")

    with gr.Blocks() as demo:
        gr.Markdown("# 🎮 Hunyuan GameCraft Video Generator")
        
        with gr.Tab("Custom Parameter Generation"):
            gr.Markdown("Upload an image or video(>=33frames) and fill in parameters to generate a video")
            
            with gr.Row():
                with gr.Column(scale=1):
                    input_content = gr.File(
                        label="Input Image or Video",
                        file_types=["image", "video"]
                    )
                    image_preview = gr.Image(label="Image Preview", visible=False, interactive=False)
                    video_preview = gr.Video(label="Video Preview", visible=False, interactive=False)
                with gr.Column(scale=1):
                    gr.Markdown("### Preset Examples (Click to load)")
                    example_gallery = gr.Gallery(
                        label="Preset Images",
                        value=[(ex["image_path"], ex["name"]) for ex in PRESET_EXAMPLES],
                        columns=3,
                        height="auto",
                        allow_preview=False
                    )
            
            input_prompt = gr.Textbox(
                value=PRESET_EXAMPLES[0]["prompt"] if PRESET_EXAMPLES else "",
                label="Prompt"
            )
            
            with gr.Row():
                action_list_input = gr.Dropdown(
                    choices=list(ACTION_MAP.keys()),
                    value=["forward", "left"],
                    multiselect=True,
                    label="Select Actions (Action List)",
                    info="You can select multiple actions, including repeated selections of the same action"
                )
                action_speed_list_input = gr.Textbox(
                    value="0.2 0.2", 
                    label="Action Speeds (Action Speed List)", 
                    info="Separated by spaces, e.g.: 0.2 0.3 0.1 (must match the number of actions)"
                )
            
            with gr.Row():
                video_continue_checkbox = gr.Checkbox(
                    value=False, 
                    label="Video Condition Continuation",
                    interactive=False,
                    info="Continue from previous video (only available after first generation and no new image/video uploaded)"
                )
                
            with gr.Accordion("Advanced Settings", open=False):
                with gr.Row():
                    pos_prompt = gr.Textbox(value="Realistic, High-quality.", label="Positive Prompt")
                    neg_prompt = gr.Textbox(
                        value="overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, text, subtitles, static, picture, black border.",
                        label="Negative Prompt"
                    )
                with gr.Row():
                    video_width = gr.Number(value=704, label="Video Width", precision=0)
                    video_height = gr.Number(value=1216, label="Video Height", precision=0)
                    cfg_scale = gr.Number(value=2.0, label="CFG Scale")
                with gr.Row():
                    seed_input = gr.Number(value=250160, label="Seed", precision=0)
                    sample_frames = gr.Number(value=33, label="Sample N Frames", precision=0)
                    infer_steps_input = gr.Number(value=50, label="Inference Steps", precision=0)
                    flow_shift = gr.Number(value=5.0, label="Flow Shift Eval Video")

            with gr.Row():
                generate_btn = gr.Button("🚀 Generate Custom Video")
            
            with gr.Row():
                video_output_custom = gr.Video(label="Generated Video")
                frame_gallery = gr.Gallery(label="Frame Images", columns=4, height="auto")
            
            with gr.Row():
                status_custom = gr.Textbox(label="Status")

            # 隐藏的状态变量，用于存储当前输入类型
            input_type_state = gr.State(value=None)

            example_gallery.select(
                fn=load_preset_by_selection,
                inputs=[],  # 占位，实际通过事件自动获取索引
                outputs=[input_content, input_prompt]
            )

            generate_btn.click(
                fn=generate_custom_video,
                inputs=[
                    input_content,
                    input_type_state,
                    video_continue_checkbox,
                    action_list_input,
                    action_speed_list_input,
                    input_prompt,
                    pos_prompt,
                    neg_prompt,
                    video_width,
                    video_height,
                    cfg_scale,
                    seed_input,
                    sample_frames,
                    infer_steps_input,
                    flow_shift
                ],
                outputs=[video_output_custom, frame_gallery, status_custom, video_continue_checkbox]
            )

            input_content.change(
                fn=handle_input_change,
                inputs=[input_content],
                outputs=[video_continue_checkbox, input_type_state]
            )

            input_content.change(
                fn=update_preview,
                inputs=[input_content],
                outputs=[image_preview, video_preview]
            )

    logger.info(f"🌐 Launching Gradio UI on port {GRADIO_PORT}...")
    
    try:
        demo.launch(server_name="0.0.0.0", server_port=GRADIO_PORT, share=True)
    except KeyboardInterrupt:
        logger.info("Gradio UI interrupted.")
    except Exception as e:
        logger.error(f"Error launching Gradio UI: {e}")
        logger.info("Falling back to local-only launch...")
        demo.launch(server_name="0.0.0.0", server_port=GRADIO_PORT, share=False)