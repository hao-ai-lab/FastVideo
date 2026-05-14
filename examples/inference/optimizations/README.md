# Optimization Examples

## Attention backend
```bash
python examples/inference/optimizations/attention_example.py
```

## Text encoder quantization (AbsMaxFP8)
```bash
python examples/inference/optimizations/text_encoder_quant_example.py \
    --text_encoder_path /path/to/umt5_xxl_fp8_e4m3fn_scaled.safetensors
```

## Text encoder quantization (GGUF)

GGUF quantization uses low-precision weight types (Q4_0, Q4_K, Q8_0, etc.) to
reduce the text encoder memory footprint while keeping quality close to the
original. Requires [vLLM](https://github.com/vllm-project/vllm) for the GGML
CUDA dequantization kernels.

```bash
python examples/inference/optimizations/gguf_quant_example.py \
    --text_encoder_path /path/to/text_encoder_gguf.safetensors
```
