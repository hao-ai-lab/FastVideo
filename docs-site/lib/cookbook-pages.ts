import sourceCatalog from "../../docs/assets/cookbook-recipes.json" with { type: "json" };

/** Presentation metadata only. Deployments and editable fields come from the cookbook backend. */
export const recipePages = [
  {
    id: "fasth3-8step",
    family: "FastH3",
    title: "FastH3 8-Step V2",
    mark: "H3",
    path: "/cookbook/minimax-h3/fasth3-8step",
    description: "Distilled MiniMax-H3 video generation with synchronized audio.",
    tags: ["Text to video", "Audio", "8-step"],
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
  serving?: { source: string; install: string; env?: string };
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
  sd35: { title: "Stable Diffusion", path: "stable-diffusion", logo: "stabilityai", category: "Image" },
  stable_audio: { title: "Stable Audio", path: "stable-audio", logo: "stabilityai", category: "Audio" },
  mmaudio: { title: "MMAudio", path: "mmaudio", logo: "fastvideo", category: "Audio" },
  cosmos: { title: "Cosmos", path: "cosmos", logo: "nvidia", category: "World" },
  matrixgame: { title: "Matrix Game", path: "matrix-game", logo: "fastvideo", category: "World" },
};
const backendModels: Record<string, string> = {
  "FastVideo/FastVideo-FastH3-8-Step-V2": "fasth3-8step",
  "FastVideo/FastWan2.1-T2V-1.3B-Diffusers": "fastwan21",
  "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers": "wan21-i2v",
  "FastVideo/LTX2-Distilled-Diffusers": "ltx2-distilled",
};
export const modelDirectory = [...new Set(sourceRecipes.map((recipe) => recipe.model))]
  .map((model) => {
    const recipes = sourceRecipes.filter((recipe) => recipe.model === model);
    const first = recipes[0];
    const family = familyInfo[first.family];
    const backend = recipePages.find((page) => page.id === backendModels[model]);
    const ownModel = model.startsWith("FastVideo/Fast");
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
      family: first.family === "minimax_h3" ? "FastH3" : (backend?.family ?? family.title),
      familyPath: `/cookbook/${family.path}`,
      path: backend?.path ?? `/cookbook/${family.path}/models/${first.id}`,
      logo: ownModel ? "fastvideo" : family.logo,
      category: family.category,
      description: backend?.description ?? first.summary,
      tags: backend?.tags ?? [...new Set(recipes.map((recipe) => recipe.task))],
      platforms: [...new Set(recipes.flatMap((recipe) => recipe.gpu_types))],
      configurable: !!backend,
      featured: backend?.featured ?? false,
    };
  })
  .sort(
    (a, b) =>
      Number(b.featured) - Number(a.featured) ||
      Number(b.family === "FastH3") - Number(a.family === "FastH3") ||
      Number(b.configurable) - Number(a.configurable),
  );
export type DirectoryModel = (typeof modelDirectory)[number];
