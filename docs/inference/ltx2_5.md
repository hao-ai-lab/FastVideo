# LTX-2.5 inference

FastVideo supports BF16 inference for the LTX-2.5 dev and distilled
transformers. Both text-to-video and image-to-video generate synchronized
video and audio. The native path supports sequence parallelism, component
offload, and `torch.compile`.

## Convert the official checkpoint

The gated [`Lightricks/LTX-2.5`](https://huggingface.co/Lightricks/LTX-2.5)
repository publishes separate transformer, packed Gemma 4, convolutional video
VAE, audio VAE/vocoder, and spatial upscaler files. Accept its license and
download those files, then convert them into one FastVideo model directory:

```bash
python scripts/checkpoint_conversion/convert_ltx2_weights.py \
  --variant distilled \
  --transformer-source /weights/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
  --text-encoder-source /weights/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
  --vae-source /weights/ltx-2.5-video-vae-conv-bf16.safetensors \
  --audio-vae-source /weights/ltx-2.5-audio-vae-bf16.safetensors \
  --spatial-upscaler-source /weights/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  --output /models/LTX-2.5-Distilled-Diffusers
```

Use `--variant dev` and the dev transformer to convert the development model.
The converter emits a standard component directory and preserves LTX-2.5's
architecture metadata, packed tokenizer, joint audio components, and refine
configuration. The convolutional video VAE is the supported launch path.

## Generate video and audio

Run distilled text-to-audio-video across four GPUs with sequence parallelism:

```bash
python examples/inference/basic/basic_ltx2_5_t2av.py \
  --model-path /models/LTX-2.5-Distilled-Diffusers \
  --prompt "A jazz trio performs in a candlelit club, synchronized live sound" \
  --num-gpus 4 \
  --torch-compile
```

Condition on a first frame with the I2AV example:

```bash
python examples/inference/basic/basic_ltx2_5_i2av.py \
  --model-path /models/LTX-2.5-Distilled-Diffusers \
  --image /images/first-frame.png \
  --prompt "The train pulls away from the platform as its horn sounds" \
  --num-gpus 4 \
  --torch-compile
```

Pass `--variant dev` with a converted dev directory to use the 30-step dev
guidance preset. Distilled inference uses the official eight-step ancestral
schedule and a three-step spatial refinement pass by default.

## Current scope

The initial inference path includes the native transformer, packed Gemma 4
text stack, convolutional video VAE, audio VAE/vocoder, dev guidance, and the
distilled ancestral sampler. DiffVAE/NATTEN, generated keyframes, temporal
upsampling, automatic duration selection, HDR, training, fine-tuning, and
quantized deployment are separate follow-up work.
