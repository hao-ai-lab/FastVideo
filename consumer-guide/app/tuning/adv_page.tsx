"use client"

import { useEffect, useId, useMemo, useState, type ElementType, type ReactNode } from "react"
import {
  BarChart3,
  Check,
  ChevronDown,
  ChevronUp,
  Copy,
  Film,
  HardDrive,
  Image,
  Info,
  Layers,
  Settings2,
  Sparkles,
  Zap,
} from "lucide-react"
import tuningData from "@/data/tuning.json"

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

interface ModelOption {
  id: string
  name: string
  size: string
  workload: string
  keyboardDim?: number
  usesMouse?: boolean
  recipe: Recipe
}

interface WorkloadOption {
  id: string
  name: string
  icon: string
}

interface AttentionBackend {
  id: string
  name: string
  shortName: string
  desc: string
}

interface Defaults {
  model_id: string
  workload_type: string
  num_gpus: number
  seed: number
  dit_cpu_offload: boolean
  dit_layerwise_offload: boolean
  vae_cpu_offload: boolean
  text_encoder_cpu_offload: boolean
  image_encoder_cpu_offload: boolean
  pin_cpu_memory: boolean
  use_fsdp_inference: boolean
}

interface Config extends Defaults {
  height: number
  width: number
  num_frames: number
  fps: number
  num_inference_steps: number
  guidance_scale: number
  guidance_scale_2: number | null
  embedded_cfg_scale: number
  boundary_ratio: number | null
  dmd_denoising_steps: number[] | null
  attention_backend: string
  VSA_sparsity: number
}

const DEFAULTS = tuningData.defaults as Defaults
const MODELS = tuningData.models as ModelOption[]
const WORKLOADS = tuningData.workloads as WorkloadOption[]
const ATTENTION_BACKENDS = tuningData.attentionBackends as AttentionBackend[]
const MODEL_BY_ID = Object.fromEntries(MODELS.map((model) => [model.id, model])) as Record<string, ModelOption>
const ATTENTION_BY_ID = Object.fromEntries(ATTENTION_BACKENDS.map((backend) => [backend.id, backend])) as Record<string, AttentionBackend>
const DEFAULT_MODEL = MODEL_BY_ID[DEFAULTS.model_id] ?? MODELS[0]
const SOURCE = tuningData.source

const WORKLOAD_ICONS: Record<string, ElementType> = {
  t2v: Film,
  i2v: Image,
  ti2v: Layers,
  game: Zap,
}

function withModel(base: Defaults | Config, model: ModelOption): Config {
  const recipe = model.recipe
  return {
    ...base,
    model_id: model.id,
    workload_type: model.workload,
    height: recipe.height,
    width: recipe.width,
    num_frames: recipe.numFrames,
    fps: recipe.fps,
    num_inference_steps: recipe.numInferenceSteps,
    guidance_scale: recipe.guidanceScale,
    guidance_scale_2: recipe.guidanceScale2,
    embedded_cfg_scale: recipe.embeddedCfgScale,
    boundary_ratio: recipe.boundaryRatio,
    dmd_denoising_steps: recipe.dmdDenoisingSteps ? [...recipe.dmdDenoisingSteps] : null,
    attention_backend: recipe.defaultAttentionBackend,
    VSA_sparsity: recipe.vsaSparsity,
  }
}

function Help({ text }: { text: string }) {
  return (
    <span className="ml-1.5 inline-flex text-slate-500" tabIndex={0} title={text} aria-label={text}>
      <Info className="h-3.5 w-3.5" aria-hidden="true" />
    </span>
  )
}

function Section({
  title,
  icon: Icon,
  children,
}: {
  title: string
  icon: ElementType
  children: ReactNode
}) {
  const [open, setOpen] = useState(true)
  const contentId = useId()
  return (
    <section className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
      <button
        type="button"
        className="flex w-full items-center justify-between p-5 text-left hover:bg-slate-50"
        aria-expanded={open}
        aria-controls={contentId}
        onClick={() => setOpen((value) => !value)}
      >
        <span className="flex items-center gap-3">
          <Icon className="h-5 w-5 text-blue-600" aria-hidden="true" />
          <span className="text-sm font-semibold uppercase tracking-wider text-slate-900">{title}</span>
        </span>
        {open ? <ChevronUp className="h-5 w-5" aria-hidden="true" /> : <ChevronDown className="h-5 w-5" aria-hidden="true" />}
      </button>
      {open && <div id={contentId} className="border-t border-slate-100 bg-slate-50/60 p-5">{children}</div>}
    </section>
  )
}

function NumberInput({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  help,
  disabled = false,
}: {
  label: string
  value: number
  onChange: (value: number) => void
  min?: number
  max?: number
  step?: number
  help?: string
  disabled?: boolean
}) {
  const id = useId()
  return (
    <div>
      <label htmlFor={id} className="mb-2 flex items-center text-xs font-semibold uppercase tracking-wide text-slate-600">
        {label}
        {help && <Help text={help} />}
      </label>
      <input
        id={id}
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        onChange={(event) => {
          let next = event.currentTarget.valueAsNumber
          if (!Number.isFinite(next)) return
          if (Number.isInteger(step)) {
            next = (min ?? 0) + Math.round((next - (min ?? 0)) / step) * step
          }
          if (min !== undefined) next = Math.max(min, next)
          if (max !== undefined) next = Math.min(max, next)
          onChange(next)
        }}
        className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5 text-sm text-slate-900 disabled:cursor-not-allowed disabled:bg-slate-100"
      />
    </div>
  )
}

function SelectInput({
  label,
  value,
  onChange,
  options,
  help,
}: {
  label: string
  value: string
  onChange: (value: string) => void
  options: { id: string; name: string }[]
  help?: string
}) {
  const id = useId()
  return (
    <div>
      <label htmlFor={id} className="mb-2 flex items-center text-xs font-semibold uppercase tracking-wide text-slate-600">
        {label}
        {help && <Help text={help} />}
      </label>
      <select
        id={id}
        value={value}
        onChange={(event) => onChange(event.currentTarget.value)}
        className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5 text-sm text-slate-900"
      >
        {options.map((option) => <option key={option.id} value={option.id}>{option.name}</option>)}
      </select>
    </div>
  )
}

function Toggle({
  label,
  checked,
  onChange,
  help,
  disabled = false,
}: {
  label: string
  checked: boolean
  onChange: (value: boolean) => void
  help?: string
  disabled?: boolean
}) {
  return (
    <div className={`flex items-center justify-between border-b border-slate-200 px-2 py-3 ${disabled ? "opacity-50" : ""}`}>
      <span className="flex items-center text-sm text-slate-700">
        {label}
        {help && <Help text={help} />}
      </span>
      <button
        type="button"
        role="switch"
        aria-label={label}
        aria-checked={checked}
        disabled={disabled}
        onClick={() => onChange(!checked)}
        className={`relative h-6 w-11 rounded-full transition-colors ${checked ? "bg-blue-600" : "bg-slate-300"}`}
      >
        <span className={`absolute left-0.5 top-0.5 h-5 w-5 rounded-full bg-white shadow transition-transform ${checked ? "translate-x-5" : ""}`} />
      </button>
    </div>
  )
}

export default function AdvancedTuningPage() {
  const [config, setConfig] = useState<Config>(() => withModel(DEFAULTS, DEFAULT_MODEL))
  const [copied, setCopied] = useState(false)

  const selectedModel = MODEL_BY_ID[config.model_id] ?? DEFAULT_MODEL
  const availableModels = useMemo(
    () => MODELS.filter((model) => model.workload === config.workload_type),
    [config.workload_type],
  )
  const availableBackends = selectedModel.recipe.attentionBackends
    .map((id) => ATTENTION_BY_ID[id])
    .filter((backend): backend is AttentionBackend => Boolean(backend))

  const updateConfig = <K extends keyof Config>(key: K, value: Config[K]) => {
    setConfig((current) => ({ ...current, [key]: value }))
  }

  const selectWorkload = (workload: string) => {
    const model = MODELS.find((candidate) => candidate.workload === workload)
    if (model) setConfig((current) => withModel(current, model))
  }

  const selectModel = (modelId: string) => {
    const model = MODEL_BY_ID[modelId]
    if (model) setConfig((current) => withModel(current, model))
  }

  const recipeChanged = useMemo(() => {
    const recipe = selectedModel.recipe
    return config.height !== recipe.height ||
      config.width !== recipe.width ||
      config.num_frames !== recipe.numFrames ||
      config.fps !== recipe.fps ||
      config.num_inference_steps !== recipe.numInferenceSteps ||
      config.guidance_scale !== recipe.guidanceScale ||
      config.guidance_scale_2 !== recipe.guidanceScale2 ||
      config.embedded_cfg_scale !== recipe.embeddedCfgScale ||
      config.boundary_ratio !== recipe.boundaryRatio ||
      config.attention_backend !== recipe.defaultAttentionBackend ||
      config.VSA_sparsity !== recipe.vsaSparsity
  }, [config, selectedModel])

  const videoLength = (config.num_frames / config.fps).toFixed(1)
  const totalPixels = config.height * config.width * config.num_frames
  const latentTokens = (Math.floor((config.num_frames - 1) / 4) + 1) *
    Math.floor(config.height / 8) * Math.floor(config.width / 8)

  useEffect(() => {
    if (window.parent === window) return
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
    const observer = new ResizeObserver(sendHeight)
    observer.observe(root)
    sendHeight()
    return () => {
      window.cancelAnimationFrame(frame)
      observer.disconnect()
    }
  }, [])

  const command = useMemo(() => {
    const isGame = config.workload_type === "game"
    const needsImage = isGame || config.workload_type === "i2v" || config.workload_type === "ti2v"
    const workloadType = config.workload_type === "ti2v" || isGame ? "i2v" : config.workload_type
    const overrideYaml = [
      "    preset_overrides:",
      `      embedded_cfg_scale: ${config.embedded_cfg_scale}`,
      ...(config.dmd_denoising_steps === null
        ? []
        : [`      dmd_denoising_steps: [${config.dmd_denoising_steps.join(", ")}]`]),
    ]

    if (isGame) {
      const pythonBoolean = (value: boolean) => value ? "True" : "False"
      const pythonOverrides = [
        ...(config.dmd_denoising_steps === null
          ? []
          : [`            "dmd_denoising_steps": [${config.dmd_denoising_steps.join(", ")}],`]),
      ]
      const inputLines = [
        '        "image_path": "./input.png",',
        selectedModel.usesMouse === false ? null : '        "mouse_cond": actions["mouse"].unsqueeze(0),',
        '        "keyboard_cond": actions["keyboard"].unsqueeze(0),',
        '        "grid_sizes": grid_sizes,',
      ].filter((line): line is string => Boolean(line))
      const keyboardDim = selectedModel.keyboardDim ?? 4
      return [
        "# Save as run.py, then run: python run.py",
        "import os",
        "import torch",
        "from fastvideo import VideoGenerator",
        "from fastvideo.models.dits.matrixgame2.utils import create_action_presets",
        "",
        `os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "${config.attention_backend}"`,
        "generator = VideoGenerator.from_config({",
        `    "model_path": "${config.model_id}",`,
        "    \"engine\": {",
        `        "num_gpus": ${config.num_gpus},`,
        `        "use_fsdp_inference": ${pythonBoolean(config.use_fsdp_inference)},`,
        "        \"offload\": {",
        `            "dit": ${pythonBoolean(config.dit_cpu_offload)},`,
        `            "dit_layerwise": ${pythonBoolean(config.dit_layerwise_offload)},`,
        `            "text_encoder": ${pythonBoolean(config.text_encoder_cpu_offload)},`,
        `            "image_encoder": ${pythonBoolean(config.image_encoder_cpu_offload)},`,
        `            "vae": ${pythonBoolean(config.vae_cpu_offload)},`,
        `            "pin_cpu_memory": ${pythonBoolean(config.pin_cpu_memory)},`,
        "        },",
        "    },",
        "    \"pipeline\": {",
        "        \"workload_type\": \"i2v\",",
        "        \"preset_overrides\": {",
        ...pythonOverrides,
        "        },",
        "    },",
        "})",
        "",
        `num_frames = ${config.num_frames}`,
        `actions = create_action_presets(num_frames, keyboard_dim=${keyboardDim}, seed=${config.seed < 0 ? "None" : config.seed})`,
        `grid_sizes = torch.tensor([${Math.floor((config.num_frames - 1) / 4) + 1}, ${Math.floor(config.height / 8)}, ${Math.floor(config.width / 8)}])`,
        "generator.generate({",
        '    "prompt": "",',
        "    \"inputs\": {",
        ...inputLines,
        "    },",
        "    \"sampling\": {",
        `        "height": ${config.height},`,
        `        "width": ${config.width},`,
        `        "num_frames": ${config.num_frames},`,
        `        "fps": ${config.fps},`,
        `        "num_inference_steps": ${config.num_inference_steps},`,
        ...(config.dmd_denoising_steps === null ? [`        "guidance_scale": ${config.guidance_scale},`] : []),
        ...(config.boundary_ratio === null ? [] : [`        "boundary_ratio": ${config.boundary_ratio},`]),
        `        "seed": ${config.seed},`,
        "    },",
        '    "output": {"output_path": "outputs/", "save_video": True},',
        "})",
      ].join("\n")
    }

    const experimentalYaml = config.attention_backend === "VIDEO_SPARSE_ATTN"
      ? ["    experimental:", `      VSA_sparsity: ${config.VSA_sparsity}`]
      : []
    const inputYaml = needsImage ? ["  inputs:", "    image_path: ./input.png"] : []

    return [
      "cat > fastvideo-generate.yaml <<'YAML'",
      "generator:",
      `  model_path: ${config.model_id}`,
      "  engine:",
      `    num_gpus: ${config.num_gpus}`,
      `    use_fsdp_inference: ${config.use_fsdp_inference}`,
      "    offload:",
      `      dit: ${config.dit_cpu_offload}`,
      `      dit_layerwise: ${config.dit_layerwise_offload}`,
      `      text_encoder: ${config.text_encoder_cpu_offload}`,
      `      image_encoder: ${config.image_encoder_cpu_offload}`,
      `      vae: ${config.vae_cpu_offload}`,
      `      pin_cpu_memory: ${config.pin_cpu_memory}`,
      "  pipeline:",
      `    workload_type: ${workloadType}`,
      ...overrideYaml,
      ...experimentalYaml,
      "",
      "request:",
      '  prompt: "your prompt here"',
      ...inputYaml,
      "  sampling:",
      `    height: ${config.height}`,
      `    width: ${config.width}`,
      `    num_frames: ${config.num_frames}`,
      `    fps: ${config.fps}`,
      `    num_inference_steps: ${config.num_inference_steps}`,
      ...(config.dmd_denoising_steps === null ? [`    guidance_scale: ${config.guidance_scale}`] : []),
      ...(config.guidance_scale_2 === null ? [] : [`    guidance_scale_2: ${config.guidance_scale_2}`]),
      ...(config.boundary_ratio === null ? [] : [`    boundary_ratio: ${config.boundary_ratio}`]),
      `    seed: ${config.seed}`,
      "  output:",
      "    output_path: outputs/",
      "    save_video: true",
      "YAML",
      "",
      `FASTVIDEO_ATTENTION_BACKEND=${config.attention_backend} \\`,
      "  fastvideo generate --config fastvideo-generate.yaml",
    ].join("\n")
  }, [config, selectedModel])

  const copyCommand = async () => {
    await navigator.clipboard.writeText(command)
    setCopied(true)
    window.setTimeout(() => setCopied(false), 2000)
  }

  const isImageWorkload = config.workload_type === "i2v" || config.workload_type === "ti2v" || config.workload_type === "game"

  return (
    <main id="config-generator-root" className="bg-white px-4 py-6 text-slate-900 sm:px-6">
      <div className="mx-auto max-w-7xl space-y-5">
        <header>
          <h1 className="text-2xl font-bold">Advanced Tuning Guide</h1>
          <p className="mt-2 max-w-3xl text-sm text-slate-600">
            Start from a maintained FastVideo recipe, then copy an executable config. Runtime and memory estimates are intentionally omitted until reproducible benchmark records are published.
          </p>
        </header>

        <div className="grid grid-cols-1 gap-3 rounded-xl border border-slate-200 bg-slate-50 p-4 text-center sm:grid-cols-3">
          <div><div className="text-xs uppercase text-slate-500">Video length</div><div className="font-mono text-sm">{videoLength}s</div></div>
          <div><div className="text-xs uppercase text-slate-500">Total pixels</div><div className="font-mono text-sm">{(totalPixels / 1_000_000).toFixed(1)}M</div></div>
          <div><div className="text-xs uppercase text-slate-500">Latent tokens</div><div className="font-mono text-sm">{latentTokens.toLocaleString()}</div></div>
        </div>

        <div className="grid grid-cols-1 gap-5 lg:grid-cols-3">
          <div className="space-y-4 lg:col-span-2">
            <Section title="Model and parallelism" icon={Settings2}>
              <div className="space-y-4">
                <div>
                  <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-600">Workload</div>
                  <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                    {WORKLOADS.map((workload) => {
                      const Icon = WORKLOAD_ICONS[workload.id] ?? Film
                      return (
                        <button
                          key={workload.id}
                          type="button"
                          aria-pressed={config.workload_type === workload.id}
                          onClick={() => selectWorkload(workload.id)}
                          className={`rounded-lg border p-3 text-left ${config.workload_type === workload.id ? "border-blue-600 bg-blue-50" : "border-slate-300 bg-white"}`}
                        >
                          <Icon className="mb-2 h-4 w-4" aria-hidden="true" />
                          <span className="text-sm font-medium">{workload.name}</span>
                        </button>
                      )
                    })}
                  </div>
                </div>
                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                  <SelectInput
                    label="Model"
                    value={config.model_id}
                    onChange={selectModel}
                    options={availableModels.map((model) => ({ id: model.id, name: `${model.name} (${model.size})` }))}
                    help="Models registered for the selected workload."
                  />
                  <NumberInput
                    label="Number of GPUs"
                    value={config.num_gpus}
                    onChange={(value) => setConfig((current) => ({
                      ...current,
                      num_gpus: value,
                      use_fsdp_inference: value > 1 && current.use_fsdp_inference,
                    }))}
                    min={1}
                    max={8}
                  />
                </div>
                <p className="text-xs text-slate-500">
                  Recipe source: <a className="underline" href={`https://github.com/hao-ai-lab/FastVideo/tree/${SOURCE.fastvideoCommit}`} target="_blank" rel="noreferrer">FastVideo {SOURCE.fastvideoCommit.slice(0, 8)}</a>. Hardware fit is not inferred.
                </p>
              </div>
            </Section>

            <Section title="Video specification" icon={Film}>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <NumberInput label="Height" value={config.height} onChange={(value) => updateConfig("height", value)} min={256} max={2160} step={8} />
                <NumberInput label="Width" value={config.width} onChange={(value) => updateConfig("width", value)} min={256} max={3840} step={8} />
                <NumberInput
                  label="Frames"
                  value={config.num_frames}
                  onChange={(value) => updateConfig("num_frames", value)}
                  min={config.workload_type === "game" ? 9 : 1}
                  max={config.workload_type === "game" ? 1197 : 1200}
                  step={config.workload_type === "game" ? 12 : 1}
                  help={config.workload_type === "game" ? "Matrix Game requires 9 + 12k frames." : undefined}
                />
                <NumberInput label="FPS" value={config.fps} onChange={(value) => updateConfig("fps", value)} min={1} max={60} />
              </div>
              {recipeChanged && <p className="mt-3 text-xs font-medium text-amber-700">Custom values differ from the maintained recipe and may be unsupported.</p>}
            </Section>

            <Section title="Quality and scheduler" icon={Sparkles}>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <NumberInput
                  label="Inference steps"
                  value={config.num_inference_steps}
                  onChange={(value) => updateConfig("num_inference_steps", value)}
                  min={1}
                  max={100}
                  disabled={config.dmd_denoising_steps !== null}
                  help={config.dmd_denoising_steps ? "This model uses the fixed schedule shown below." : "Number of denoising iterations."}
                />
                {config.workload_type !== "game" && (
                  <NumberInput
                    label="Guidance scale"
                    value={config.guidance_scale}
                    onChange={(value) => updateConfig("guidance_scale", value)}
                    min={0.5}
                    max={20}
                    step={0.5}
                    disabled={config.dmd_denoising_steps !== null}
                    help={config.dmd_denoising_steps ? "DMD uses the embedded CFG scale instead." : "Values above 1 may add an unconditional forward pass."}
                  />
                )}
                {config.guidance_scale_2 !== null && <NumberInput label="Guidance scale 2" value={config.guidance_scale_2} onChange={(value) => updateConfig("guidance_scale_2", value)} min={0.5} max={20} step={0.5} />}
                {config.workload_type !== "game" && <NumberInput label="Embedded CFG scale" value={config.embedded_cfg_scale} onChange={(value) => updateConfig("embedded_cfg_scale", value)} min={0} max={20} step={0.5} />}
                {config.boundary_ratio !== null && <NumberInput label="Boundary ratio" value={config.boundary_ratio} onChange={(value) => updateConfig("boundary_ratio", value)} min={0} max={1} step={0.025} />}
                <NumberInput label="Seed" value={config.seed} onChange={(value) => updateConfig("seed", value)} min={0} />
              </div>
              {config.dmd_denoising_steps && (
                <div className="mt-4 rounded-lg bg-blue-50 p-3 text-sm text-blue-900">
                  Fixed denoising schedule: <code>{config.dmd_denoising_steps.join(", ")}</code>
                </div>
              )}
            </Section>

            <Section title="Attention" icon={Zap}>
              <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                {availableBackends.map((backend) => (
                  <button
                    key={backend.id}
                    type="button"
                    aria-pressed={config.attention_backend === backend.id}
                    onClick={() => updateConfig("attention_backend", backend.id)}
                    className={`rounded-lg border p-3 text-left ${config.attention_backend === backend.id ? "border-blue-600 bg-blue-50" : "border-slate-300 bg-white"}`}
                  >
                    <div className="font-medium">{backend.name}</div>
                    <div className="mt-1 text-xs text-slate-500">{backend.desc}</div>
                  </button>
                ))}
              </div>
              {config.attention_backend === "VIDEO_SPARSE_ATTN" && (
                <div className="mt-4 max-w-xs">
                  <NumberInput label="VSA sparsity" value={config.VSA_sparsity} onChange={(value) => updateConfig("VSA_sparsity", value)} min={0} max={1} step={0.1} />
                </div>
              )}
            </Section>

            <Section title="Memory" icon={HardDrive}>
              <Toggle
                label="dit_cpu_offload"
                checked={config.dit_cpu_offload}
                onChange={(value) => setConfig((current) => ({ ...current, dit_cpu_offload: value, dit_layerwise_offload: value ? false : current.dit_layerwise_offload }))}
              />
              <Toggle
                label="dit_layerwise_offload"
                checked={config.dit_layerwise_offload}
                onChange={(value) => setConfig((current) => ({ ...current, dit_layerwise_offload: value, dit_cpu_offload: value ? false : current.dit_cpu_offload, use_fsdp_inference: value ? false : current.use_fsdp_inference }))}
                help="Mutually exclusive with whole-DiT offload and FSDP."
              />
              <Toggle label="vae_cpu_offload" checked={config.vae_cpu_offload} onChange={(value) => updateConfig("vae_cpu_offload", value)} />
              <Toggle label="text_encoder_cpu_offload" checked={config.text_encoder_cpu_offload} onChange={(value) => updateConfig("text_encoder_cpu_offload", value)} />
              <Toggle label="image_encoder_cpu_offload" checked={config.image_encoder_cpu_offload} onChange={(value) => updateConfig("image_encoder_cpu_offload", value)} disabled={!isImageWorkload} />
              <Toggle label="pin_cpu_memory" checked={config.pin_cpu_memory} onChange={(value) => updateConfig("pin_cpu_memory", value)} />
              <Toggle label="use_fsdp_inference" checked={config.use_fsdp_inference} onChange={(value) => updateConfig("use_fsdp_inference", value)} disabled={config.num_gpus <= 1 || config.dit_layerwise_offload} />
            </Section>
          </div>

          <aside className="space-y-4">
            <div className="overflow-hidden rounded-xl border border-blue-200 bg-white shadow-sm lg:sticky lg:top-4">
              <div className="flex items-center justify-between border-b border-blue-100 bg-blue-50 p-4">
                <div className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wider text-blue-700">
                  <Settings2 className="h-4 w-4" aria-hidden="true" />
                  Generated config
                </div>
                <button type="button" onClick={copyCommand} className="flex items-center gap-2 rounded-lg px-3 py-2 text-sm hover:bg-blue-100">
                  {copied ? <Check className="h-4 w-4" aria-hidden="true" /> : <Copy className="h-4 w-4" aria-hidden="true" />}
                  {copied ? "Copied" : "Copy"}
                </button>
              </div>
              <pre className="max-h-[70vh] overflow-auto whitespace-pre-wrap bg-slate-950 p-4 font-mono text-xs leading-relaxed text-slate-100">{command}</pre>
            </div>

            <div className="rounded-xl border border-slate-200 bg-white p-4">
              <div className="mb-3 flex items-center gap-2 text-xs font-semibold uppercase text-slate-600">
                <BarChart3 className="h-4 w-4" aria-hidden="true" />
                Recipe summary
              </div>
              <dl className="space-y-2 text-xs">
                <div className="flex justify-between gap-3"><dt>Model</dt><dd className="text-right font-mono">{selectedModel.name}</dd></div>
                <div className="flex justify-between"><dt>Resolution</dt><dd className="font-mono">{config.width}×{config.height}</dd></div>
                <div className="flex justify-between"><dt>Frames</dt><dd className="font-mono">{config.num_frames}</dd></div>
                <div className="flex justify-between"><dt>Backend</dt><dd className="font-mono">{config.attention_backend}</dd></div>
              </dl>
            </div>
          </aside>
        </div>
      </div>
    </main>
  )
}
