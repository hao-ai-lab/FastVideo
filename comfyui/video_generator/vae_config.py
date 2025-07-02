class VAEConfig:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "load_encoder": ([True, False], {
                    "default": True
                }),
                "load_decoder": ([True, False], {
                    "default": True
                }),
                "tile_sample_min_height": ("INT", {
                    "default": 256
                }),
                "tile_sample_min_width": ("INT", {
                    "default": 256
                }),
                "tile_sample_min_num_frames": ("INT", {
                    "default": 16
                }),
                "tile_sample_stride_height": ("INT", {
                    "default": 192
                }),
                "tile_sample_stride_width": ("INT", {
                    "default": 192
                }),
                "tile_sample_stride_num_frames": ("INT", {
                    "default": 12
                }),
                "blend_num_frames": ("INT", {
                    "default": 0
                }),
                "use_tiling": ([True, False], {
                    "default": True
                }),
                "use_temporal_tiling": ([True, False], {
                    "default": True
                }),
                "use_parallel_tiling": ([True, False], {
                    "default": True
                }),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, scale_factor=None, **kwargs):
        return True

        # Handle None value for scale_factor
        # if scale_factor is None:
        #     # This is valid - we'll use the default value in the set_args method
        #     return True

        # For non-None values, ensure it's a valid integer
        # try:
        #     int(scale_factor)
        #     return True
        # except (ValueError, TypeError):
        #     return f"scale_factor must be an integer, got {type(scale_factor).__name__}"

    RETURN_TYPES = ("VAE_CONFIG", )
    RETURN_NAMES = ("vae_config", )
    FUNCTION = "set_args"
    CATEGORY = "fastvideo"

    def set_args(
        self,
        load_encoder,
        load_decoder,
        tile_sample_min_height,
        tile_sample_min_width,
        tile_sample_min_num_frames,
        tile_sample_stride_height,
        tile_sample_stride_width,
        tile_sample_stride_num_frames,
        blend_num_frames,
        use_tiling,
        use_temporal_tiling,
        use_parallel_tiling,
    ):
        raw_args = {
            "load_encoder": load_encoder,
            "load_decoder": load_decoder,
            "tile_sample_min_height": tile_sample_min_height,
            "tile_sample_min_width": tile_sample_min_width,
            "tile_sample_min_num_frames": tile_sample_min_num_frames,
            "tile_sample_stride_height": tile_sample_stride_height,
            "tile_sample_stride_width": tile_sample_stride_width,
            "tile_sample_stride_num_frames": tile_sample_stride_num_frames,
            "blend_num_frames": blend_num_frames,
            "use_tiling": use_tiling,
            "use_temporal_tiling": use_temporal_tiling,
            "use_parallel_tiling": use_parallel_tiling,
        }

        # Filter out any value explicitly set to -99999
        args = {k: v for k, v in raw_args.items() if str(int(v)) != str(-99999)}

        return (args, )
