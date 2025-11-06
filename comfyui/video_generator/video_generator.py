from __future__ import annotations

import glob
import os
import signal
import sys
import threading
import time
from typing import Any

from comfy.model_management import processing_interrupted

from fastvideo import PipelineConfig
from fastvideo import VideoGenerator as FastVideoGenerator

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


# Custom exception for interruption
class GenerationInterruptedException(Exception):
    pass


# Custom exception for interruption that ComfyUI will recognize
class GenerationCancelledException(Exception):

    def __init__(self,
                 message: str = "Generation was cancelled by user") -> None:
        self.message = message
        super().__init__(self.message)


def update_config_from_args(config: Any, args_dict: dict[str, Any]) -> None:
    """
    Update configuration object from arguments dictionary.
    
    Args:
        config: The configuration object to update
        args_dict: Dictionary containing arguments
    """
    for key, value in args_dict.items():
        if hasattr(config, key) and value is not None:
            if key == "text_encoder_precisions" and isinstance(value, list):
                setattr(config, key, tuple(value))
            else:
                setattr(config, key, value)


class VideoGenerator:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline":
                    True,
                    "default":
                    "A ripe orange tumbles gently from a tree and lands on the head of a lounging capybara, "
                    "who blinks slowly in response. The moment is quietly humorous and oddly serene, framed by "
                    "lush green foliage and dappled sunlight. Mid-shot, warm and whimsical tones."
                }),
                "output_path": ("STRING", {
                    "default": "/workspace/ComfyUI/outputs_video/"
                }),
                "num_gpus": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 16
                }),
                "model_path": ("STRING", {
                    "default": "FastVideo/FastHunyuan-diffusers"
                })
            },
            "optional": {
                "inference_args": ("INFERENCE_ARGS", ),
                "embedded_cfg_scale": ("FLOAT", {
                    "default": 6.0
                }),
                "sp_size": ("INT", {
                    "default": 2
                }),
                "tp_size": ("INT", {
                    "default": 2
                }),
                "vae_config": ("VAE_CONFIG", ),
                "vae_precision": (["fp16", "bf16"], {
                    "default": "fp16"
                }),
                "vae_tiling": ([True, False], {
                    "default": True
                }),
                "vae_sp": ([True, False], {
                    "default": False
                }),
                "text_encoder_config": ("TEXT_ENCODER_CONFIG", ),
                "text_encoder_precision": (["fp16", "bf16"], {
                    "default": "fp16"
                }),
                "dit_config": ("DIT_CONFIG", ),
                "precision": (["fp16", "bf16"], {
                    "default": "fp16"
                }),
                "dit_cpu_offload": ([True, False], {
                    "default": False
                }),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("video_path", )
    FUNCTION = "launch_inference"
    CATEGORY = "fastvideo"

    generator: FastVideoGenerator | None = None
    _interrupt_thread: threading.Thread | None = None
    _generation_active: bool = False
    _generation_interrupted: bool = False
    _interrupt_event: threading.Event = threading.Event()
    _generation_thread: threading.Thread | None = None
    _generation_result: str | None = None
    _generation_exception: Exception | None = None

    def _monitor_for_interruption(self):
        """Background thread that monitors for interruption requests"""
        time.sleep(2)  # Give the generation thread time to send execute_forward

        while self._generation_active and not self._interrupt_event.is_set():
            if processing_interrupted():
                print("Video generation interrupted by user")
                self._generation_interrupted = True

                # Try to send interrupt signal to worker processes
                if self.generator is not None and hasattr(
                        self.generator, 'executor'):
                    try:
                        # The MultiprocExecutor has a workers attribute
                        if hasattr(self.generator.executor, 'workers'):
                            for worker in self.generator.executor.workers:
                                if worker.is_alive():
                                    os.kill(worker.pid, signal.SIGINT)
                            print("Interrupt signal sent to worker processes")
                    except Exception as e:
                        print(f"Error sending interrupt signal: {e}")

                # Set the interrupt event to notify other threads
                self._interrupt_event.set()
                break
            time.sleep(0.5)

    def _run_generation(self, prompt: str, output_path: str,
                        inference_args: dict[str, Any]) -> None:
        """Thread function to run the generation"""
        try:
            if self.generator is not None:
                self.generator.generate_video(prompt=prompt,
                                              output_path=output_path,
                                              **inference_args)
                self._generation_result = os.path.join(output_path,
                                                       f"{prompt[:100]}.mp4")
            else:
                raise RuntimeError("Generator is not initialized")
        except Exception as e:
            self._generation_exception = e
            self._interrupt_event.set()

    def load_output_video(self, output_dir):
        video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
        video_files = []

        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(output_dir, ext)))

        if not video_files:
            print("No video files found in output directory: %s", output_dir)
            return ""

        video_files.sort()
        return video_files[0]

    def launch_inference(
        self,
        prompt,
        output_path,
        num_gpus,
        model_path,
        embedded_cfg_scale,
        sp_size,
        tp_size,
        vae_precision,
        vae_tiling,
        vae_sp,
        text_encoder_precision,
        precision,
        inference_args=None,
        vae_config=None,
        text_encoder_config=None,
        dit_config=None,
        dit_cpu_offload=None,
    ):
        print('Running FastVideo inference')

        # Reset interruption flag and event
        self._generation_interrupted = False
        self._interrupt_event.clear()
        self._generation_result = None
        self._generation_exception = None

        # Load pipeline config from model path
        pipeline_config = PipelineConfig.from_pretrained(model_path)
        print('pipeline_config', pipeline_config)

        # Update configs with provided config dictionaries
        if dit_config is not None:
            update_config_from_args(pipeline_config.dit_config, dit_config)

        if vae_config is not None:
            update_config_from_args(pipeline_config.vae_config, vae_config)

        if text_encoder_config is not None:
            update_config_from_args(pipeline_config.text_encoder_configs,
                                    text_encoder_config)

        # Update top-level pipeline config with remaining arguments
        raw_pipeline_args = {}
        if embedded_cfg_scale is not None:
            raw_pipeline_args['embedded_cfg_scale'] = embedded_cfg_scale
        if precision is not None:
            raw_pipeline_args['precision'] = precision
        if vae_precision is not None:
            raw_pipeline_args['vae_precision'] = vae_precision
        if vae_tiling is not None:
            raw_pipeline_args['vae_tiling'] = vae_tiling
        if vae_sp is not None:
            raw_pipeline_args['vae_sp'] = vae_sp
        if text_encoder_precision is not None:
            raw_pipeline_args['text_encoder_precision'] = text_encoder_precision

        # Filter out any value explicitly set to -99999 (auto values)
        pipeline_args = {
            k: v
            for k, v in raw_pipeline_args.items() if str(int(v)) != str(-99999)
        }

        update_config_from_args(pipeline_config, pipeline_args)

        raw_generation_args = {}
        if num_gpus is not None:
            raw_generation_args['num_gpus'] = num_gpus
        if tp_size is not None:
            raw_generation_args['tp_size'] = tp_size
        if sp_size is not None:
            raw_generation_args['sp_size'] = sp_size
        if dit_cpu_offload is not None:
            raw_generation_args['dit_cpu_offload'] = dit_cpu_offload

        generation_args = {
            k: v
            for k, v in raw_generation_args.items()
            if str(int(v)) != str(-99999)
        }

        if self.generator is None:
            print('generation_args', generation_args)
            print('pipeline_config', pipeline_config)
            self.generator = FastVideoGenerator.from_pretrained(
                model_path=model_path,
                **generation_args,
                pipeline_config=pipeline_config)

        print('inference_args', inference_args)

        # Start a thread to run the generation
        self._generation_thread = threading.Thread(target=self._run_generation,
                                                   args=(prompt, output_path,
                                                         inference_args),
                                                   daemon=True)
        self._generation_thread.start()

        # Start a background thread to monitor for interruptions
        self._generation_active = True
        self._interrupt_thread = threading.Thread(
            target=self._monitor_for_interruption, daemon=True)
        self._interrupt_thread.start()

        # Wait for either completion or interruption
        while self._generation_thread.is_alive(
        ) and not self._interrupt_event.is_set():
            self._generation_thread.join(timeout=0.5)

        self._generation_active = False
        if self._interrupt_thread:
            self._interrupt_thread.join(timeout=1.0)
            self._interrupt_thread = None

        if self._generation_interrupted:
            print("Video generation was cancelled by user")
            raise GenerationCancelledException()
        elif self._generation_exception:
            # Re-raise the exception from the generation thread
            raise self._generation_exception
        elif self._generation_result:
            return (self._generation_result, )
        else:
            # This shouldn't happen, but just in case
            print("Generation completed but no result was produced")
            raise Exception("Generation failed to produce a result")
