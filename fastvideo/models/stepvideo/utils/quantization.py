# from stepvideo.diffusion.video_pipeline import StepVideoPipeline
import torch
import torch.nn as nn
from torch.nn import functional as F


def get_fp_maxval(bits=8, mantissa_bit=3, sign_bits=1):
    _bits = torch.tensor(bits)
    _mantissa_bit = torch.tensor(mantissa_bit)
    _sign_bits = torch.tensor(sign_bits)
    M = torch.clamp(torch.round(_mantissa_bit), 1, _bits - _sign_bits)
    E = _bits - _sign_bits - M
    bias = 2**(E - 1) - 1
    mantissa = 1
    for i in range(mantissa_bit - 1):
        mantissa += 1 / (2**(i + 1))
    maxval = mantissa * 2**(2**E - 1 - bias)
    return maxval


def quantize_to_fp8(x, bits=8, mantissa_bit=3, sign_bits=1):
    """
    Default is E4M3.
    """
    bits = torch.tensor(bits)
    mantissa_bit = torch.tensor(mantissa_bit)
    sign_bits = torch.tensor(sign_bits)
    M = torch.clamp(torch.round(mantissa_bit), 1, bits - sign_bits)
    E = bits - sign_bits - M
    bias = 2**(E - 1) - 1
    mantissa = 1
    for i in range(mantissa_bit - 1):
        mantissa += 1 / (2**(i + 1))
    maxval = mantissa * 2**(2**E - 1 - bias)
    minval = -maxval
    minval = -maxval if sign_bits == 1 else torch.zeros_like(maxval)
    input_clamp = torch.min(torch.max(x, minval), maxval)
    log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(input_clamp)) + bias)).detach(), 1.0)
    log_scales = 2.0**(log_scales - M - bias.type(x.dtype))
    # dequant
    qdq_out = torch.round(input_clamp / log_scales) * log_scales
    return qdq_out, log_scales


def fp8_tensor_quant(x, scale, bits=8, mantissa_bit=3, sign_bits=1):
    for i in range(len(x.shape) - 1):
        scale = scale.unsqueeze(-1)
    new_x = x / scale
    quant_dequant_x, log_scales = quantize_to_fp8(new_x, bits=bits, mantissa_bit=mantissa_bit, sign_bits=sign_bits)
    return quant_dequant_x, scale, log_scales


def fp8_activation_dequant(qdq_out, scale, dtype):
    qdq_out = qdq_out.type(dtype)
    quant_dequant_x = qdq_out * scale.to(dtype)
    return quant_dequant_x


def fp8_linear_forward(cls, original_dtype, input):
    weight_dtype = cls.weight.dtype
    #####
    if cls.weight.dtype != torch.float8_e4m3fn:
        assert False
        maxval = get_fp_maxval()
        scale = torch.max(torch.abs(cls.weight.flatten())) / maxval
        linear_weight, scale, log_scales = fp8_tensor_quant(cls.weight, scale)
        linear_weight = linear_weight.to(torch.float8_e4m3fn)
        weight_dtype = linear_weight.dtype
    else:
        scale = cls.fp8_scale.to(cls.weight.device)
        linear_weight = cls.weight
    #####

    if weight_dtype == torch.float8_e4m3fn:
        if True or len(input.shape) == 3:
            cls_dequant = fp8_activation_dequant(linear_weight, scale, original_dtype)
            if cls.bias is not None:
                print(f"input dtype: {input.dtype}")
                print(f"cls_dequant dtype: {cls_dequant.dtype}")
                print(f"cls.bias dtype: {cls.bias.dtype}")

                output = F.linear(input, cls_dequant, cls.bias)
            else:
                output = F.linear(input, cls_dequant)
            return output
        else:
            return cls.original_forward(input.to(original_dtype))
    else:
        return cls.original_forward(input)


def convert_fp8_linear(module, original_dtype, params_to_keep={}):
    setattr(module, "fp8_matmul_enabled", True)
    fp8_layers = []
    scale_dict = {}
    counter = 0
    for key, layer in module.named_modules():
        if isinstance(layer, nn.Linear) and 'transformer_blocks' in key:
            print(f"Converting {key} to FP8")
            fp8_layers.append(key)
            original_forward = layer.forward
            maxval = get_fp_maxval()
            scale = torch.max(torch.abs(layer.weight.flatten())) / maxval

            original_weight = layer.weight.data  # Store a reference to the original weights
            quantized_weight, scale, _ = fp8_tensor_quant(original_weight, scale)
            scale_dict[key] = scale
            layer.weight = torch.nn.Parameter(quantized_weight.to(torch.float8_e4m3fn))
            del original_weight  # Delete the reference to the original weights
            torch.cuda.empty_cache()

            # print(f"layer weight dtype: {layer.weight.dtype} for layer {key}")
            setattr(layer, "fp8_scale", scale.to(dtype=original_dtype))
            setattr(layer, "original_forward", original_forward)
            setattr(layer, "forward", lambda input, m=layer: fp8_linear_forward(m, original_dtype, input))
            counter += 1
    return scale_dict
