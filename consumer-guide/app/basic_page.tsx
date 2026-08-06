"use client"

import { useEffect, useState, useMemo } from "react"
import { Layers, Copy, Check, Film, Image, Sparkles, Gamepad2 } from "lucide-react"
import quickstartData from "@/data/quickstart.json"
import tuningData from "@/data/tuning.json"

type IconName = "Film" | "Image" | "Sparkles" | "Gamepad2"

interface Recipe {
  height: number
  width: number
  numFrames: number
  fps: number
  numInferenceSteps: number
  guidanceScale: number
  guidanceScale2: number | null
  embeddedCfgScale: number
  boundaryRatio: number | null
  dmdDenoisingSteps: number[] | null
  attentionBackends: string[]
  defaultAttentionBackend: string
  vsaSparsity: number
}

interface TaskOption {
  id: string
  label: string
  sub: string
  icon: string
}

interface ModelOption {
  id: string
  name: string
  workload: string
  keyboardDim?: number
  usesMouse?: boolean
  recipe: Recipe
}

interface TierOption {
  id: string
  name: string
  desc: string
  badge?: string
}

interface Profile {
  task: string
  tier: string
  model: string
  ditCpuOffload: boolean
  ditLayerwiseOffload: boolean
  vaeCpuOffload: boolean
  textEncoderCpuOffload: boolean
  imageEncoderCpuOffload: boolean
  pinCpuMemory: boolean
  useFsdpInference: boolean
}

interface QuickstartDefaults {
  task: string
  tier: string
  numGpus: number
}

const ICONS: Record<IconName, typeof Film> = {
  Film,
  Image,
  Sparkles,
  Gamepad2,
}

const DEFAULTS = quickstartData.defaults as QuickstartDefaults
const GPU_COUNT_OPTIONS = quickstartData.gpuCountOptions as number[]
const PROFILES = quickstartData.profiles as Profile[]
const TASKS = quickstartData.tasks as TaskOption[]
const TIERS = quickstartData.tiers as TierOption[]
const MODELS = tuningData.models as ModelOption[]
const MODEL_BY_ID = Object.fromEntries(MODELS.map((model) => [model.id, model])) as Record<string, ModelOption>

function getProfile(task: string, tier: string): Profile | undefined {
  return PROFILES.find((profile) => profile.task === task && profile.tier === tier)
}

export default function FastVideoConfigSelector() {
  const [task, setTask] = useState(DEFAULTS.task)
  const [tier, setTier] = useState(DEFAULTS.tier)
  const [ngpu, setNgpu] = useState(DEFAULTS.numGpus)
  const [copied, setCopied] = useState(false)

  const profile = getProfile(task, tier)
  const selectedModel = profile ? MODEL_BY_ID[profile.model] : undefined
  const config = profile && selectedModel ? { ...profile, ...selectedModel.recipe, ...selectedModel } : undefined

  // Report content height to the parent docs page so the embedding iframe can
  // auto-resize — same mechanism as the Advanced Tuning guide.
  useEffect(() => {
    if (typeof window === "undefined" || window.parent === window) return
    const root = document.getElementById("config-generator-root")
    if (!root) return

    let frame = 0
    const sendHeight = () => {
      window.cancelAnimationFrame(frame)
      frame = window.requestAnimationFrame(() => {
        window.parent.postMessage(
          { type: "fastvideo-config-generator:resize", height: Math.ceil(root.getBoundingClientRect().height) },
          window.location.origin,
        )
      })
    }

    const ro = new ResizeObserver(sendHeight)
    ro.observe(root)
    sendHeight()

    return () => {
      window.cancelAnimationFrame(frame)
      ro.disconnect()
    }
  }, [])

  const command = useMemo(() => {
    if (!config) return ""

    const workloadType = task === "ti2v" ? "i2v" : task
    const pythonBoolean = (value: boolean) => value ? "True" : "False"
    const presetOverrideEntries = [
      task === "game" ? null : `["embedded_cfg_scale", ${config.embeddedCfgScale}]`,
      config.dmdDenoisingSteps === null
        ? null
        : `["dmd_denoising_steps", [${config.dmdDenoisingSteps.join(", ")}]]`,
    ].filter((entry): entry is string => Boolean(entry))

    if (task === "game") {
      const keyboardDim = config.keyboardDim ?? 4
      const inputLines = [
        '        "image_path": "./input.png",',
        config.usesMouse === false ? null : '        "mouse_cond": actions["mouse"].unsqueeze(0),',
        '        "keyboard_cond": actions["keyboard"].unsqueeze(0),',
        '        "grid_sizes": grid_sizes,',
      ].filter((line): line is string => Boolean(line))
      return [
        "# Game control uses the Python API because mouse/keyboard",
        "# conditions are tensors, not simple CLI fields.",
        "",
        "# Install once:",
        "#   pip install fastvideo",
        "",
        "# Save this file as run.py and run:",
        "#   python run.py",
        "import os",
        "import torch",
        "from fastvideo import VideoGenerator",
        "from fastvideo.models.dits.matrixgame2.utils import create_action_presets",
        "",
        `os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "${config.defaultAttentionBackend}"`,
        "",
        "generator = VideoGenerator.from_config({",
        `    "model_path": "${config.model}",`,
        "    \"engine\": {",
        `        "num_gpus": ${ngpu},`,
        `        "use_fsdp_inference": ${pythonBoolean(config.useFsdpInference)},`,
        "        \"offload\": {",
        `            "dit": ${pythonBoolean(config.ditCpuOffload)},`,
        `            "dit_layerwise": ${pythonBoolean(config.ditLayerwiseOffload)},`,
        `            "text_encoder": ${pythonBoolean(config.textEncoderCpuOffload)},`,
        `            "image_encoder": ${pythonBoolean(config.imageEncoderCpuOffload)},`,
        `            "vae": ${pythonBoolean(config.vaeCpuOffload)},`,
        `            "pin_cpu_memory": ${pythonBoolean(config.pinCpuMemory)},`,
        "        },",
        "    },",
        "    \"pipeline\": {",
        "        \"workload_type\": \"i2v\",",
        "        \"preset_overrides\": dict([",
        ...presetOverrideEntries.map((entry) => `            ${entry},`),
        "        ]),",
        "    },",
        "})",
        "",
        `num_frames = ${config.numFrames}`,
        `actions = create_action_presets(num_frames, keyboard_dim=${keyboardDim}, seed=1024)`,
        `grid_sizes = torch.tensor([${Math.floor((config.numFrames - 1) / 4) + 1}, ${Math.floor(config.height / 8)}, ${Math.floor(config.width / 8)}])`,
        "",
        "generator.generate({",
        '    "prompt": "",',
        "    \"inputs\": {",
        ...inputLines,
        "    },",
        "    \"sampling\": {",
        `        "height": ${config.height},`,
        `        "width": ${config.width},`,
        `        "fps": ${config.fps},`,
        "        \"seed\": 1024,",
        `        "num_frames": ${config.numFrames},`,
        `        "num_inference_steps": ${config.numInferenceSteps},`,
        ...(config.dmdDenoisingSteps === null ? [`        "guidance_scale": ${config.guidanceScale},`] : []),
        ...(config.boundaryRatio === null ? [] : [`        "boundary_ratio": ${config.boundaryRatio},`]),
        "    },",
        '    "output": {"output_path": "outputs/", "save_video": True},',
        "})",
      ].join("\n")
    }

    const offloadYaml = [
      `      dit: ${config.ditCpuOffload}`,
      `      dit_layerwise: ${config.ditLayerwiseOffload}`,
      `      text_encoder: ${config.textEncoderCpuOffload}`,
      `      image_encoder: ${config.imageEncoderCpuOffload}`,
      `      vae: ${config.vaeCpuOffload}`,
      `      pin_cpu_memory: ${config.pinCpuMemory}`,
    ]
    const inputsYaml = task === "i2v" || task === "ti2v"
      ? [
          "  inputs:",
          "    image_path: ./input.png",
        ]
      : []
    const presetOverridesYaml = [
      "    preset_overrides:",
      `      embedded_cfg_scale: ${config.embeddedCfgScale}`,
      ...(config.dmdDenoisingSteps === null
        ? []
        : [`      dmd_denoising_steps: [${config.dmdDenoisingSteps.join(", ")}]`]),
    ]
    const experimentalYaml = config.defaultAttentionBackend === "VIDEO_SPARSE_ATTN"
      ? ["    experimental:", `      VSA_sparsity: ${config.vsaSparsity}`]
      : []

    return [
      "# Install once:",
      "#   pip install fastvideo",
      "",
      "cat > fastvideo-generate.yaml <<'YAML'",
      "generator:",
      `  model_path: ${config.model}`,
      "  engine:",
      `    num_gpus: ${ngpu}`,
      `    use_fsdp_inference: ${config.useFsdpInference}`,
      "    offload:",
      ...offloadYaml,
      "  pipeline:",
      `    workload_type: ${workloadType}`,
      ...presetOverridesYaml,
      ...experimentalYaml,
      "",
      "request:",
      '  prompt: "your prompt here"',
      ...inputsYaml,
      "  sampling:",
      `    height: ${config.height}`,
      `    width: ${config.width}`,
      `    fps: ${config.fps}`,
      "    seed: 1024",
      `    num_frames: ${config.numFrames}`,
      `    num_inference_steps: ${config.numInferenceSteps}`,
      ...(config.dmdDenoisingSteps === null ? [`    guidance_scale: ${config.guidanceScale}`] : []),
      ...(config.guidanceScale2 === null ? [] : [`    guidance_scale_2: ${config.guidanceScale2}`]),
      ...(config.boundaryRatio === null ? [] : [`    boundary_ratio: ${config.boundaryRatio}`]),
      "  output:",
      "    output_path: outputs/",
      "    save_video: true",
      "YAML",
      "",
      `FASTVIDEO_ATTENTION_BACKEND=${config.defaultAttentionBackend} \\`,
      "  fastvideo generate --config fastvideo-generate.yaml",
    ].join("\n")
  }, [config, ngpu, task])

  const copyCommand = async () => {
    if (!command) return
    await navigator.clipboard.writeText(command)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div id="config-generator-root" className="bg-white text-foreground">
      <div className="max-w-4xl px-6 py-8">
        {/* Step 1: Task Selection */}
        <section className="mb-8">
          <div className="mb-4 flex items-center gap-2">
            <span className="flex h-6 w-6 items-center justify-center rounded-full bg-primary text-xs font-medium text-primary-foreground">1</span>
            <h2 className="text-sm font-semibold uppercase tracking-wider text-primary">What do you want to make?</h2>
          </div>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            {TASKS.map(({ id, label, sub, icon }) => {
              const Icon = ICONS[icon as IconName] ?? Film
              return (
                <button
                  type="button"
                  key={id}
                  onClick={() => setTask(id)}
                  aria-pressed={task === id}
                  className={`group relative rounded-xl border p-4 text-left transition-all duration-200 hover:border-primary/50 ${
                    task === id
                      ? "border-primary bg-primary/10 shadow-lg shadow-primary/10"
                      : "border-border bg-card hover:bg-card/80"
                  }`}
                >
                  <div className={`mb-3 flex h-10 w-10 items-center justify-center rounded-lg transition-colors ${
                    task === id ? "bg-primary text-primary-foreground" : "bg-secondary text-muted-foreground"
                  }`}>
                    <Icon className="h-5 w-5" />
                  </div>
                  <div className="font-medium text-card-foreground">
                    {label}
                  </div>
                  <div className="mt-0.5 text-xs text-muted-foreground">{sub}</div>
                </button>
              )
            })}
          </div>
        </section>

        {/* Step 2: Parallelism */}
        <section className="mb-8">
          <div className="mb-4 flex items-center gap-2">
            <span className="flex h-6 w-6 items-center justify-center rounded-full bg-primary text-xs font-medium text-primary-foreground">2</span>
            <h2 className="text-sm font-semibold uppercase tracking-wider text-primary">Parallelism</h2>
          </div>
          <div className="max-w-xs rounded-xl border border-border bg-card p-4">
              <label htmlFor="quick-gpu-count" className="mb-2 flex items-center gap-2 text-primary">
                <Layers className="h-4 w-4" />
                <span className="text-xs font-semibold uppercase tracking-wide">Number of GPUs</span>
              </label>
              <select
                id="quick-gpu-count"
                value={ngpu}
                onChange={(e) => setNgpu(Number(e.target.value))}
                className="w-full rounded-lg border border-border bg-card px-3 py-2 text-sm text-card-foreground outline-none transition-colors focus:border-primary/50"
              >
                {GPU_COUNT_OPTIONS.map((option) => (
                  <option key={option} value={option}>{option} GPU{option === 1 ? "" : "s"}</option>
                ))}
              </select>
          </div>
        </section>

        {/* Step 3: Model profile */}
        <section className="mb-8">
          <div className="mb-4 flex items-center gap-2">
            <span className="flex h-6 w-6 items-center justify-center rounded-full bg-primary text-xs font-medium text-primary-foreground">3</span>
            <h2 className="text-sm font-semibold uppercase tracking-wider text-primary">Model profile</h2>
          </div>
          <div className="grid gap-3 sm:grid-cols-4">
            {TIERS.map(({ id, name, desc, badge }) => {
              const tierConfig = getProfile(task, id)
              const tierModel = tierConfig ? MODEL_BY_ID[tierConfig.model] : undefined
              return (
                <button
                  type="button"
                  key={id}
                  onClick={() => setTier(id)}
                  aria-pressed={tier === id}
                  className={`group relative rounded-xl border p-4 text-left transition-all duration-200 sm:col-span-2 ${
                    tier === id
                      ? "border-primary bg-primary/10 shadow-lg shadow-primary/10"
                      : "border-border bg-card hover:border-primary/50 hover:bg-card/80"
                  }`}
                >
                  {badge && (
                    <span className="mb-2 inline-block rounded bg-primary/20 px-2 py-0.5 text-xs font-medium text-primary">
                      {badge}
                    </span>
                  )}
                  <div className="font-medium text-card-foreground">
                    {name}
                  </div>
                  <p className="mt-1 text-xs leading-relaxed text-muted-foreground">{desc}</p>
                  <div className="mt-3 space-y-1 border-t border-border pt-3">
                    {tierModel ? (
                      <div className="text-xs text-muted-foreground">
                        <span className="font-semibold">{tierModel.name}</span>
                        {` · ${tierModel.recipe.width}×${tierModel.recipe.height} · ${tierModel.recipe.numInferenceSteps} steps`}
                      </div>
                    ) : (
                      <div className="text-xs font-medium text-muted-foreground">
                        Not suitable
                      </div>
                    )}
                  </div>
                </button>
              )
            })}
          </div>
        </section>

        {/* Output Configuration */}
        <section>
          <div className="mb-4 flex items-center gap-2">
            <span className="flex h-6 w-6 items-center justify-center rounded-full bg-primary text-xs font-medium text-primary-foreground">
              <Check className="h-4 w-4" />
            </span>
            <h2 className="text-sm font-semibold uppercase tracking-wider text-primary">Your configuration</h2>
          </div>
          <div className="overflow-hidden rounded-2xl border border-border bg-card">
            {/* Meta Cards */}
            <div className="grid grid-cols-1 divide-y divide-border border-b border-border sm:grid-cols-3 sm:divide-x sm:divide-y-0">
              <div className="p-4">
                <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Model</div>
                <div className="mt-1 font-mono text-sm text-card-foreground">{config?.name ?? "Unavailable"}</div>
              </div>
              <div className="p-4">
                <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Resolution</div>
                <div className="mt-1 font-mono text-sm text-card-foreground">{config ? `${config.width}×${config.height}` : "Unavailable"}</div>
              </div>
              <div className="p-4">
                <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Denoising</div>
                <div className="mt-1 font-mono text-sm text-card-foreground">{config ? `${config.numInferenceSteps} steps` : "Unavailable"}</div>
              </div>
            </div>
            {/* Command Block */}
            <div className="relative p-4">
              {command && (
                <button
                  type="button"
                  onClick={copyCommand}
                  className="absolute right-4 top-4 flex items-center gap-2 rounded-lg border border-border bg-card px-4 py-2 text-sm font-medium text-muted-foreground transition-all hover:border-primary/50 hover:bg-primary/10 hover:text-primary"
                >
                  {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                  {copied ? "Copied" : "Copy"}
                </button>
              )}
              {command ? (
                <pre className="overflow-x-auto font-mono text-sm leading-relaxed text-muted-foreground">
                  {command}
                </pre>
              ) : (
                <div className="rounded-lg border border-dashed border-border px-4 py-6 text-sm text-muted-foreground">
                  No maintained profile is available for this task and model choice.
                </div>
              )}
            </div>
          </div>
        </section>
      </div>
    </div>
  )
}
