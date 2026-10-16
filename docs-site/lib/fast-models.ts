export const families = [
  {
    name: "FastH3",
    mark: "H3",
    summary: "Distilled MiniMax-H3 video with synchronized audio.",
    collection: "https://huggingface.co/collections/FastVideo/fasth3-v2",
    guide: "/cookbook/minimax-h3",
    detail: "8-step V2 \u00b7 FP8 and NVFP4 exports",
    models: [
      {
        id: "FastVideo-FastH3-8-Step-V2",
        url: "https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2",
      },
      {
        id: "FastVideo-FastH3-8-Step-V2-NVFP4",
        url: "https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4",
      },
      {
        id: "FastVideo-FastH3-8-Step-V2-FP8",
        url: "https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2-FP8",
      },
      {
        id: "FastVideo-FastH3-8-Step-V2-NVFP4-Consumer",
        url: "https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4-Consumer",
      },
      {
        id: "FastVideo-FastH3-Trim-8-Step",
        url: "https://huggingface.co/FastVideo/FastVideo-FastH3-Trim-8-Step",
      },
      {
        id: "FastVideo-FastH3-Trim-8-Step-NVFP4",
        url: "https://huggingface.co/FastVideo/FastVideo-FastH3-Trim-8-Step-NVFP4",
      },
      {
        id: "FastVideo-FastH3-Trim-8-Step-FP8",
        url: "https://huggingface.co/FastVideo/FastVideo-FastH3-Trim-8-Step-FP8",
      },
      {
        id: "FastVideo-FastH3-8-Step-V2-NVFP4-Comfy",
        url: "https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2-NVFP4-Comfy",
      },
      {
        id: "FastVideo-FastH3-Trim-Comfy",
        url: "https://huggingface.co/FastVideo/FastVideo-FastH3-Trim-Comfy",
      },
      {
        id: "FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree",
        url: "https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree",
      },
    ],
  },
  {
    name: "FastWan",
    mark: "Wan",
    summary: "Distilled Wan models for text-to-video and image-to-video.",
    collection: "https://huggingface.co/collections/FastVideo/fastwan",
    guide: "/cookbook/wan",
    detail: "1.3B, 5B, and 14B releases",
    models: [
      {
        id: "FastWan2.1-T2V-1.3B-Diffusers",
        url: "https://huggingface.co/FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
      },
      {
        id: "FastWan2.1-T2V-14B-Diffusers",
        url: "https://huggingface.co/FastVideo/FastWan2.1-T2V-14B-Diffusers",
      },
      {
        id: "FastWan2.2-TI2V-5B-FullAttn-Diffusers",
        url: "https://huggingface.co/FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers",
      },
      {
        id: "FastWan-QAD-FP8-1.3B",
        url: "https://huggingface.co/FastVideo/FastWan-QAD-FP8-1.3B",
      },
      {
        id: "CausalWan2.2-I2V-A14B-Preview-Diffusers",
        url: "https://huggingface.co/FastVideo/CausalWan2.2-I2V-A14B-Preview-Diffusers",
      },
    ],
  },
  {
    name: "FastMetal",
    mark: "Metal",
    summary: "Quantized Wan checkpoints for Apple Silicon and MLX.",
    collection: "https://huggingface.co/collections/FastVideo/fastmetal",
    guide: "/getting_started/installation/mlx",
    detail: "INT8 QAD \u00b7 1.3B, 5B, and 14B",
    models: [
      {
        id: "FastMetal-1.3B-QAD",
        url: "https://huggingface.co/FastVideo/FastMetal-1.3B-QAD",
      },
      {
        id: "FastMetal-5B-QAD",
        url: "https://huggingface.co/FastVideo/FastMetal-5B-QAD",
      },
      {
        id: "FastMetal-14B-QAD",
        url: "https://huggingface.co/FastVideo/FastMetal-14B-QAD",
      },
    ],
  },
];
