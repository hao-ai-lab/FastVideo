class InferenceArgs:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "height": ("INT", {
                    "default": 720
                }),
                "width": ("INT", {
                    "default": 1280
                }),
                "num_frames": ("INT", {
                    "default": 45
                }),
                "num_inference_steps": ("INT", {
                    "default": 6
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 1.0
                }),
                "flow_shift": ("INT", {
                    "default": 17
                }),
                "seed": ("INT", {
                    "default": 1024
                }),
                "fps": ("INT", {
                    "default": 24
                }),
                "image_path": ("STRING", {
                    "default": "X://insert/path/here.mp4"
                }),
                "enable_teacache": ([True, False], {
                    "default": False
                }),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    RETURN_TYPES = ("INFERENCE_ARGS", )
    RETURN_NAMES = ("inference_args", )
    FUNCTION = "set_args"
    CATEGORY = "fastvideo"

    def set_args(
        self,
        height,
        width,
        num_frames,
        num_inference_steps,
        guidance_scale,
        flow_shift,
        seed,
        fps,
        image_path,
        enable_teacache,
    ):
        raw_args = {
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "flow_shift": flow_shift,
            "seed": seed,
            "fps": fps,
            "image_path": image_path,
            "enable_teacache": enable_teacache,
        }

        # Filter out keys where value is -99999, handling different types properly
        args = {}
        for k, v in raw_args.items():
            try:
                if isinstance(v, str):
                    if v != "-99999":
                        args[k] = v
                elif v != -99999:
                    # If it's not a string, compare directly
                    args[k] = v
            except (ValueError, TypeError):
                # Include any value that causes an error in comparison
                args[k] = v

        return (args, )
