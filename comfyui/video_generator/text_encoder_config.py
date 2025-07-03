class TextEncoderConfig:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "prefix": ("STRING", {
                    "default": ""
                }),
                "quant_config": ("STRING", {
                    "default": ""
                }),
                "lora_config": ("STRING", {
                    "default": ""
                }),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    RETURN_TYPES = ("TEXT_ENCODER_CONFIG", )
    RETURN_NAMES = ("text_encoder_config", )
    FUNCTION = "set_args"
    CATEGORY = "fastvideo"

    def set_args(self, prefix, quant_config, lora_config):
        raw_args = {
            "prefix": prefix,
            "quant_config": quant_config,
            "lora_config": lora_config
        }

        # Filter out keys where value is -99999
        args = {k: v for k, v in raw_args.items() if str(int(v)) != str(-99999)}

        return (args, )
