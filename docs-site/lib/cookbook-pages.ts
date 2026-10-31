import sourceCatalog from "../../docs/assets/cookbook-recipes.json" with { type: "json" };

/** Presentation metadata only. Deployments and editable fields come from the cookbook backend. */
export const recipePages = [
  {
    id: "fasth3-preview",
    family: "MiniMax H3",
    title: "FastH3 Preview V0.2",
    mark: "H3",
    path: "/cookbook/minimax-h3/fasth3-preview",
    description: "The few-step FastH3 preview checkpoint on CUDA or native Apple Silicon MLX.",
    tags: ["Text to video", "Audio", "4-step"],
  },
  {
    id: "fasth3-8step",
    family: "MiniMax H3",
    title: "FastH3 8-Step V2",
    mark: "H3",
    path: "/cookbook/minimax-h3/fasth3-8step",
    description: "Distilled MiniMax-H3 video generation with synchronized audio.",
    tags: ["Text to video", "Audio", "8-step"],
  },
  {
    id: "fasth3-trim",
    family: "MiniMax H3",
    title: "FastH3 Trim",
    mark: "H3",
    path: "/cookbook/minimax-h3/fasth3-trim",
    description: "The smaller FastH3 model for eight-step video generation with synchronized audio.",
    tags: ["Text to video", "Audio", "NVFP4", "8-step"],
  },
  {
    id: "fasth3-4step",
    family: "MiniMax H3",
    title: "FastH3 4-Step V1",
    mark: "H3",
    path: "/cookbook/minimax-h3/fasth3-4step",
    description: "FastH3 V1 on one DGX Spark with offloaded encoders and lazy module loading.",
    tags: ["Text to video", "Audio", "DGX Spark"],
  },
  {
    id: "fasth3-omniref",
    family: "MiniMax H3",
    title: "FastH3 Omni v1",
    mark: "H3",
    path: "/cookbook/minimax-h3/fasth3-omniref",
    description: "Video and audio from a prompt plus ordered image, video, and audio references.",
    tags: ["Reference to video", "Audio", "NVFP4"],
    featured: true,
  },
  {
    id: "fastwan21",
    family: "Wan",
    title: "FastWan2.1 1.3B",
    mark: "Wan",
    path: "/cookbook/wan/fastwan21",
    description: "A compact, distilled Wan model for text-to-video generation.",
    tags: ["Text to video", "1.3B", "3-step"],
  },
  {
    id: "wan21-i2v",
    family: "Wan",
    title: "Wan2.1 I2V 14B",
    mark: "Wan",
    path: "/cookbook/wan/wan21-i2v",
    description: "Turn a reference image into video with Wan2.1.",
    tags: ["Image to video", "14B", "480p"],
  },
  {
    id: "ltx2-distilled",
    family: "LTX",
    title: "LTX2 Distilled",
    mark: "LTX",
    path: "/cookbook/ltx/ltx2-distilled",
    description: "Generate video with incremental delivery over WebSocket.",
    tags: ["Text to video", "Streaming"],
  },
];
export type RecipePage = (typeof recipePages)[number];
export function sitePath(path: string) {
  return `${import.meta.env.BASE_URL.replace(/\/$/, "")}${path}`;
}

export interface SourceRecipe {
  id: string;
  family: string;
  task: string;
  label: string;
  summary: string;
  model: string;
  source: string;
  command: string;
  gpu_types: string[];
  hardware?: { gpu_count?: number; platform?: string; accelerator?: string };
  serving?: { source: string; install: string; env?: string; prepare?: string };
  knobs?: {
    key: string;
    label: string;
    hint?: string;
    flag: string;
    default: string | number;
    options: (string | number | { value: string | number; label: string })[];
  }[];
}
export const sourceRecipes = sourceCatalog.recipes as SourceRecipe[];
const familyInfo: Record<string, { title: string; path: string; logo: string; category: string }> = {
  minimax_h3: { title: "MiniMax H3", path: "minimax-h3", logo: "minimax", category: "Video" },
  wan: { title: "Wan", path: "wan", logo: "wan-ai", category: "Video" },
  ltx2: { title: "LTX", path: "ltx", logo: "ltx", category: "Video" },
  hunyuan: { title: "Hunyuan", path: "hunyuan", logo: "tencent-hunyuan", category: "Video" },
  kandinsky5: { title: "Kandinsky 5", path: "kandinsky5", logo: "kandinsky", category: "Video" },
  kandinsky6: { title: "Kandinsky 6", path: "kandinsky6", logo: "kandinsky", category: "Video" },
  flux: { title: "FLUX", path: "flux", logo: "black-forest-labs", category: "Image" },
  qwen_image21: { title: "Qwen-Image-2.1", path: "qwen-image21", logo: "qwen", category: "Image" },
  sd35: { title: "Stable Diffusion", path: "stable-diffusion", logo: "stabilityai", category: "Image" },
  stable_audio: { title: "Stable Audio", path: "stable-audio", logo: "stabilityai", category: "Audio" },
  mmaudio: { title: "MMAudio", path: "mmaudio", logo: "fastvideo", category: "Audio" },
  cosmos: { title: "Cosmos", path: "cosmos", logo: "nvidia", category: "World" },
  matrixgame: { title: "Matrix Game", path: "matrix-game", logo: "fastvideo", category: "World" },
};
const backendModels: Record<string, string> = {
  "FastVideo/FastVideo-FastH3-8-Step-V2": "fasth3-8step",
  "FastVideo/FastVideo-FastH3-Trim-8-Step-NVFP4": "fasth3-trim",
  "FastVideo/FastVideo-Minimax-FastH3-Preview-v0.2": "fasth3-preview",
  "FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree": "fasth3-4step",
  "FastVideo/FastH3-OmniRef-NVFP4-QAD": "fasth3-omniref",
  "FastVideo/FastWan2.1-T2V-1.3B-Diffusers": "fastwan21",
  "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers": "wan21-i2v",
  "FastVideo/LTX2-Distilled-Diffusers": "ltx2-distilled",
};
const hiddenBackends = new Set(["fasth3-preview"]);

const modelAddedOn: Record<string, string> = {
  "Qwen/Qwen-Image-2.1": "2026-10-25",
};
export function recentlyAdded(model: string) {
  const added = modelAddedOn[model];
  return !!added && Date.now() - Date.parse(added) < 30 * 86_400_000;
}

const TASK_GROUPS: Array<[RegExp, string]> = [
  [/world/, "Text to world"],
  [/video\/text to audio|video.*to audio/, "Video to audio"],
  [/text to audio/, "Text to audio"],
  [/text to image/, "Text to image"],
  [/image to image/, "Image to image"],
  [/reference media to video/, "Reference to video"],
  [/first\/last frame to video/, "Image to video"],
  [/text or image to video/, "Text/image to video"],
  [/image to video/, "Image to video"],
  [/super-resolution/, "Video upscale"],
  [/video/, "Text to video"],
];
/** A small discovery vocabulary derived from the recipe task, not a capability claim. */
export function taskLabel(value: string): string {
  const normalized = value.toLowerCase();
  return TASK_GROUPS.find(([pattern]) => pattern.test(normalized))?.[1] ?? "Text to video";
}

const mappedModels = [...new Set(sourceRecipes.map((recipe) => recipe.model))].map((model) => {
  const recipes = sourceRecipes.filter((recipe) => recipe.model === model);
  const first = recipes[0];
  const family = familyInfo[first.family];
  const backend = recipePages.find((page) => page.id === backendModels[model]);
  const ownModel = model.startsWith("FastVideo/Fast");
  const logo = ownModel ? "fastvideo" : family.logo;
  const providers: Record<string, string> = {
    fastvideo: "FastVideo",
    minimax: "MiniMax",
    "wan-ai": "Wan AI",
    ltx: "Lightricks",
    "tencent-hunyuan": "Tencent",
    kandinsky: "Kandinsky Lab",
    "black-forest-labs": "Black Forest Labs",
    qwen: "Qwen",
    stabilityai: "Stability AI",
    nvidia: "NVIDIA",
  };
  return {
    id: backend?.id ?? first.id,
    model,
    recipes,
    title:
      backend?.title ??
      (model.includes("FastH3-4-step")
        ? "FastH3 4-Step V1"
        : model.includes("FastH3-Preview-v0.2")
          ? "FastH3 Preview V0.2"
          : model === "MiniMaxAI/MiniMax-H3"
            ? "MiniMax H3"
            : first.label.split(" on ")[0]),
    family: backend?.family ?? family.title,
    familyPath: `/cookbook/${family.path}`,
    path: backend?.path ?? `/cookbook/${family.path}/models/${first.id}`,
    logo,
    provider: providers[logo],
    category: family.category,
    description: backend?.description ?? first.summary,
    tags: backend?.tags ?? [...new Set(recipes.map((recipe) => recipe.task))],
    task: taskLabel(backend?.tags[0] ?? first.task),
    platforms: [...new Set(recipes.flatMap((recipe) => recipe.gpu_types))],
    configurable: !!backend,
    hidden: hiddenBackends.has(backend?.id ?? ""),
    featured: backend?.featured ?? false,
  };
});

const byPage = new Map<string, (typeof mappedModels)[number]>();
for (const entry of mappedModels) {
  const existing = byPage.get(entry.id);
  if (!existing) {
    byPage.set(entry.id, entry);
    continue;
  }
  existing.recipes.push(...entry.recipes.filter((recipe) => !existing.recipes.some((seen) => seen.id === recipe.id)));
  existing.platforms = [...new Set([...existing.platforms, ...entry.platforms])];
}

export const modelDirectory = [...byPage.values()].sort(
  (a, b) =>
    Number(b.featured) - Number(a.featured) ||
    Number(b.family === "MiniMax H3") - Number(a.family === "MiniMax H3") ||
    Number(b.configurable) - Number(a.configurable),
);
export type DirectoryModel = (typeof modelDirectory)[number];
