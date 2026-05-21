"use client"

import { useState, useMemo, useEffect, type CSSProperties } from "react"
import {
  Monitor, Cpu, Film, Sparkles, Zap, HardDrive, BarChart3,
  Copy, Check, Info, ChevronDown, ChevronUp, Settings2,
  Layers, Clock, Database, Gauge
} from "lucide-react"
import tuningData from "@/data/tuning.json"

// Types
interface Config {
  // Category A - Model & Hardware
  model_id: string
  workload_type: string
  gpu: string
  num_gpus: number
  // Category B - Video Spec
  height: number
  width: number
  num_frames: number
  fps: number
  // Category C - Quality
  num_inference_steps: number
  guidance_scale: number
  embedded_cfg_scale: number
  flow_shift: number | null
  seed: number
  // Category D - Speed
  attention_backend: string
  VSA_sparsity: number
  dmd_denoising_steps: string
  // Category E - Memory
  dit_cpu_offload: boolean
  dit_layerwise_offload: boolean
  vae_cpu_offload: boolean
  text_encoder_cpu_offload: boolean
  image_encoder_cpu_offload: boolean
  pin_cpu_memory: boolean
  use_fsdp_inference: boolean
}

const SECTION_THEMES = {
  sky: {
    accent: "#3867f0",
    soft: "#e9eefc",
    tint: "#f8faff",
    border: "#c9d5fb",
  },
  emerald: {
    accent: "#089da8",
    soft: "#e7f7f8",
    tint: "#f4fbfb",
    border: "#9ddde2",
  },
  amber: {
    accent: "#f25a00",
    soft: "#fff0e8",
    tint: "#fff8f4",
    border: "#ffc6a6",
  },
  violet: {
    accent: "#00983f",
    soft: "#e9f8ef",
    tint: "#f6fcf8",
    border: "#9edeb7",
  },
  rose: {
    accent: "#a84ac2",
    soft: "#f6edf9",
    tint: "#fcf7fd",
    border: "#dfb6eb",
  },
  slate: {
    accent: "#475569",
    soft: "#f1f5f9",
    tint: "#f8fafc",
    border: "#cbd5e1",
  },
}

interface BenchmarkRun {
  model: string
  gpu: string
  height: number
  width: number
  frames: number
  steps: number
  attention: string
  generationSeconds?: number
  runtimeSeconds?: number
}

interface ModelOption {
  id: string
  name: string
  size: string
  workload: string
  keyboardDim?: number
  dmdMode?: boolean
  dmdNote?: string
}

interface WorkloadOption {
  id: string
  name: string
  icon: string
}

interface GpuOption {
  id: string
  name: string
  vram: string
}

interface ResolutionPreset {
  height: number
  width: number
  label: string
}

interface FramePreset {
  frames: number
  label: string
}

interface AttentionBackend {
  id: string
  name: string
  shortName: string
  desc: string
  speed: string
}

const modelOptions = tuningData.models as ModelOption[]
const workloadOptions = tuningData.workloads as WorkloadOption[]
const gpuOptions = tuningData.gpus as GpuOption[]
const resolutionPresets = tuningData.resolutionPresets as ResolutionPreset[]
const framePresets = tuningData.framePresets as FramePreset[]
const attentionBackends = tuningData.attentionBackends as AttentionBackend[]
const BENCHMARK_RUNS = tuningData.runs as BenchmarkRun[]
const MODEL_BY_ID = Object.fromEntries(modelOptions.map((model) => [model.id, model])) as Record<string, (typeof modelOptions)[number]>
const ATTENTION_BY_ID = Object.fromEntries(attentionBackends.map((backend) => [backend.id, backend])) as Record<string, AttentionBackend>

// Tooltip component
function Tooltip({ content, side = "top", iconClassName }: { content: string; side?: "top" | "right" | "bottom"; iconClassName?: string }) {
  const [show, setShow] = useState(false)
  const tooltipClass =
    side === "right"
      ? "absolute left-full top-1/2 z-50 ml-2 w-72 -translate-y-1/2 rounded-lg border border-slate-200 bg-white p-3 text-xs text-slate-600 shadow-xl"
      : side === "bottom"
      ? "absolute top-full left-1/2 z-50 mt-2 w-64 -translate-x-1/2 rounded-lg border border-slate-200 bg-white p-3 text-xs text-slate-600 shadow-xl"
      : "absolute bottom-full left-1/2 z-50 mb-2 w-64 -translate-x-1/2 rounded-lg border border-slate-200 bg-white p-3 text-xs text-slate-600 shadow-xl"
  const arrowClass =
    side === "right"
      ? "absolute right-full top-1/2 -translate-y-1/2 border-4 border-transparent border-r-white"
      : side === "bottom"
      ? "absolute left-1/2 bottom-full -translate-x-1/2 border-4 border-transparent border-b-white"
      : "absolute left-1/2 top-full -translate-x-1/2 border-4 border-transparent border-t-white"

  return (
    <div className="relative inline-block">
      <button
        onMouseEnter={() => setShow(true)}
        onMouseLeave={() => setShow(false)}
        className={iconClassName ?? "ml-1.5 text-slate-500 transition-colors hover:text-[var(--section-color)]"}
      >
        <Info className="h-3.5 w-3.5" />
      </button>
      {show && (
        <div className={tooltipClass}>
          {content}
          <div className={arrowClass} />
        </div>
      )}
    </div>
  )
}

// Collapsible section
function Section({
  title,
  icon: Icon,
  number,
  children,
  defaultOpen = true,
  color = "sky"
}: {
  title: string
  icon: React.ElementType
  number: string
  children: React.ReactNode
  defaultOpen?: boolean
  color?: "sky" | "emerald" | "amber" | "violet" | "rose" | "slate"
}) {
  const [open, setOpen] = useState(defaultOpen)
  const theme = SECTION_THEMES[color]
  const sectionStyle = {
    "--section-color": theme.accent,
    "--section-soft": theme.soft,
    "--section-tint": theme.tint,
    "--section-border": theme.border,
  } as CSSProperties

  return (
    <div className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm" style={sectionStyle}>
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-center justify-between p-5 transition-colors hover:bg-slate-50"
      >
        <div className="flex items-center gap-3">
          <span className="flex h-8 w-8 items-center justify-center rounded-full bg-[var(--section-color)] text-sm font-semibold text-white">
            {number}
          </span>
          <Icon className="h-5 w-5 text-[var(--section-color)]" />
          <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-900">{title}</h2>
        </div>
        {open ? <ChevronUp className="h-5 w-5 text-slate-600" /> : <ChevronDown className="h-5 w-5 text-slate-600" />}
      </button>
      {open && <div className="border-t border-slate-100 bg-[var(--section-tint)] p-6">{children}</div>}
    </div>
  )
}

// Input components
function NumberInput({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  tooltip,
  unit
}: {
  label: string
  value: number
  onChange: (v: number) => void
  min?: number
  max?: number
  step?: number
  tooltip?: string
  unit?: string
}) {
  return (
    <div>
      <label className="mb-2 flex items-center text-xs font-semibold uppercase tracking-wide text-slate-600">
        {label}
        {tooltip && <Tooltip content={tooltip} />}
      </label>
      <div className="relative">
        <input
          type="number"
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          min={min}
          max={max}
          step={step}
          className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2.5 text-sm text-slate-900 shadow-sm focus:border-[var(--section-color)] focus:outline-none focus:ring-2 focus:ring-[var(--section-color)]/20"

        />
        {unit && <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-white/80">{unit}</span>}
      </div>
    </div>
  )
}

function SelectInput({
  label,
  value,
  onChange,
  options,
  tooltip,
  disabled = false
}: {
  label: string
  value: string
  onChange: (v: string) => void
  options: { id: string; name: string; desc?: string }[]
  tooltip?: string
  disabled?: boolean
}) {
  return (
    <div>
      <label className="mb-2 flex items-center text-xs font-semibold uppercase tracking-wide text-slate-600">
        {label}
        {tooltip && <Tooltip content={tooltip} />}
      </label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        className={`w-full rounded-lg border border-slate-300 px-3 py-2.5 text-sm text-slate-900 shadow-sm focus:border-[var(--section-color)] focus:outline-none focus:ring-2 focus:ring-[var(--section-color)]/20 ${
          disabled ? "cursor-not-allowed bg-slate-100 text-slate-500" : "bg-white"
        }`}
      >
        {options.map((opt) => (
          <option key={opt.id} value={opt.id}>{opt.name}</option>
        ))}
      </select>
    </div>
  )
}

function Toggle({
  label,
  checked,
  onChange,
  tooltip,
  disabled = false
}: {
  label: string
  checked: boolean
  onChange: (v: boolean) => void
  tooltip?: string
  disabled?: boolean
}) {
  return (
    <label className={`flex cursor-pointer items-center justify-between border-b border-slate-200 px-3 py-4 transition-colors hover:bg-white/70 ${disabled ? 'cursor-not-allowed opacity-50' : ''}`}>
      <div className="flex items-center gap-2">
        <span className="text-sm uppercase tracking-wide text-slate-600">{label}</span>
        {tooltip && <Tooltip content={tooltip} />}
      </div>
      <button
        type="button"
        disabled={disabled}
        onClick={() => !disabled && onChange(!checked)}
        className={`relative h-6 w-11 rounded-full transition-colors ${checked ? 'bg-[var(--section-color)]' : 'bg-slate-200'}`}
      >
        <span className={`absolute top-0.5 left-0.5 h-5 w-5 rounded-full bg-white shadow transition-transform ${checked ? 'translate-x-5' : ''}`} />
      </button>
    </label>
  )
}

export default function AdvancedTuningPage() {
  const [copied, setCopied] = useState(false)
  const [config, setConfig] = useState<Config>(tuningData.defaults as Config)

  const videoLength = useMemo(() => (config.num_frames / config.fps).toFixed(1), [config.num_frames, config.fps])
  const totalPixels = useMemo(() => config.height * config.width * config.num_frames, [config.height, config.width, config.num_frames])

  // LATENT TOKENS — the sequence length the DiT actually processes. The VAE
  // compresses the video 4x in time and 8x8 in space; the DiT flattens that
  // latent grid into a token sequence. Attention cost scales with its square.
  const latentTokens = useMemo(() => {
    const t = Math.floor((config.num_frames - 1) / 4) + 1
    const h = Math.floor(config.height / 8)
    const w = Math.floor(config.width / 8)
    return t * h * w
  }, [config.num_frames, config.height, config.width])

  // EST. TIME — real measured generation time, looked up from data/tuning.json
  // by model + GPU + attention backend. There is no formula: the tuning page allows
  // arbitrary configs, but only the benchmarked points have a real time. If the user
  // tunes resolution / frames / steps away from the benchmarked run, the value is
  // flagged as no longer matching their config.
  const benchmark = useMemo(() => {
    const run = BENCHMARK_RUNS.find(
      (r) =>
        r.model === config.model_id &&
        MODEL_BY_ID[r.model]?.workload === config.workload_type &&
        r.gpu === config.gpu &&
        r.attention === config.attention_backend,
    )
    if (!run) return null
    const deviated =
      config.height !== run.height ||
      config.width !== run.width ||
      config.num_frames !== run.frames ||
      config.num_inference_steps !== run.steps
    return { run, deviated }
  }, [
    config.model_id,
    config.workload_type,
    config.gpu,
    config.attention_backend,
    config.height,
    config.width,
    config.num_frames,
    config.num_inference_steps,
  ])

  // When the model or GPU changes, snap the time-affecting parameters to the
  // values that model + GPU was actually benchmarked at, so the page defaults to
  // a real measured configuration. No-op if that model + GPU has no benchmark.
  useEffect(() => {
    const run =
      BENCHMARK_RUNS.find(
        (r) =>
          r.model === config.model_id &&
          MODEL_BY_ID[r.model]?.workload === config.workload_type &&
          r.gpu === config.gpu &&
          r.attention === config.attention_backend,
      ) ??
      BENCHMARK_RUNS.find(
        (r) =>
          r.model === config.model_id &&
          MODEL_BY_ID[r.model]?.workload === config.workload_type &&
          r.gpu === config.gpu,
      )
    if (!run) return
    setConfig((prev) => ({
      ...prev,
      height: run.height,
      width: run.width,
      num_frames: run.frames,
      num_inference_steps: run.steps,
      attention_backend: run.attention,
    }))
  }, [config.model_id, config.workload_type, config.gpu])

  // The Model dropdown offers successful benchmark profiles for the currently
  // selected Workload Type + GPU. If the same model has multiple successful
  // attention backends, each benchmarked backend is listed separately.
  const availableModelRuns = useMemo(
    () =>
      BENCHMARK_RUNS.filter(
        (run) =>
          MODEL_BY_ID[run.model]?.workload === config.workload_type &&
          run.gpu === config.gpu,
      ),
    [config.workload_type, config.gpu],
  )

  // Keep the selected model valid when Workload Type / GPU changes.
  useEffect(() => {
    if (availableModelRuns.length === 0) return
    if (!availableModelRuns.some((run) => run.model === config.model_id && run.attention === config.attention_backend)) {
      setConfig((prev) => ({
        ...prev,
        model_id: availableModelRuns[0].model,
        attention_backend: availableModelRuns[0].attention,
      }))
    }
  }, [availableModelRuns, config.model_id, config.attention_backend])

  const hasAvailableModelRuns = availableModelRuns.length > 0
  const selectedModelRun = `${config.model_id}::${config.attention_backend}`

  const command = useMemo(() => {
    const isGame = config.workload_type === "game"
    const isI2VLocal = config.workload_type === "i2v" || config.workload_type === "ti2v"

    // Engine kwargs: forwarded directly into VideoGenerator.from_pretrained(...).
    // Only the subset listed in fastvideo/entrypoints/video_generator.py
    // (_FROM_PRETRAINED_CONVENIENCE_KWARGS) is supported there.
    const engineKwargs: string[] = [`    num_gpus=${config.num_gpus},`]
    if (config.dit_cpu_offload) engineKwargs.push(`    dit_cpu_offload=True,`)
    if (config.dit_layerwise_offload) engineKwargs.push(`    dit_layerwise_offload=True,`)
    if (config.vae_cpu_offload) engineKwargs.push(`    vae_cpu_offload=True,`)
    if (config.text_encoder_cpu_offload) engineKwargs.push(`    text_encoder_cpu_offload=True,`)
    if (config.image_encoder_cpu_offload && isI2VLocal) {
      engineKwargs.push(`    image_encoder_cpu_offload=True,`)
    }
    if (config.pin_cpu_memory) engineKwargs.push(`    pin_cpu_memory=True,`)
    if (config.use_fsdp_inference && config.num_gpus > 1) {
      engineKwargs.push(`    use_fsdp_inference=True,`)
    }

    if (isGame) {
      const keyboardDim = MODEL_BY_ID[config.model_id]?.keyboardDim ?? 4
      return [
        "# Install once:",
        "#   pip install fastvideo",
        "",
        "# Save this file as run.py and run:",
        "#   python run.py",
        "import os",
        "import torch",
        "from fastvideo import VideoGenerator",
        "from fastvideo.models.dits.matrixgame.utils import create_action_presets",
        "",
        `os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "${config.attention_backend}"`,
        "",
        "generator = VideoGenerator.from_pretrained(",
        `    "${config.model_id}",`,
        ...engineKwargs,
        ")",
        "",
        `num_frames = ${config.num_frames}`,
        `actions = create_action_presets(num_frames, keyboard_dim=${keyboardDim})`,
        `grid_sizes = torch.tensor([${Math.floor((config.num_frames - 1) / 4) + 1}, ${Math.floor(config.height / 8)}, ${Math.floor(config.width / 8)}])`,
        "",
        "generator.generate_video(",
        '    prompt="",',
        '    image_path="./input.png",',
        '    mouse_cond=actions["mouse"].unsqueeze(0),',
        '    keyboard_cond=actions["keyboard"].unsqueeze(0),',
        "    grid_sizes=grid_sizes,",
        `    height=${config.height},`,
        `    width=${config.width},`,
        `    fps=${config.fps},`,
        "    seed=42,",
        `    num_frames=${config.num_frames},`,
        `    num_inference_steps=${config.num_inference_steps},`,
        `    guidance_scale=${config.guidance_scale},`,
        '    output_path="outputs/",',
        "    save_video=True,",
        ")",
      ].join("\n")
    }

    // SamplingParam fields: real attribute names from fastvideo/api/sampling_param.py.
    const paramAssigns: string[] = [
      `param.height = ${config.height}`,
      `param.width = ${config.width}`,
      `param.num_frames = ${config.num_frames}`,
      `param.fps = ${config.fps}`,
      `param.num_inference_steps = ${config.num_inference_steps}`,
      `param.guidance_scale = ${config.guidance_scale}`,
    ]
    if (config.seed !== -1) paramAssigns.push(`param.seed = ${config.seed}`)
    if (isI2VLocal) paramAssigns.push(`param.image_path = "./input.png"`)

    // Pipeline-config-level overrides (embedded_cfg_scale, flow_shift, VSA_sparsity)
    // are not part of from_pretrained convenience kwargs. They live in the model's
    // pipeline config JSON (see fastvideo/configs/*.json). Surface as comment so users
    // know they need a custom config file to apply them.
    const pipelineNotes: string[] = []
    if (config.embedded_cfg_scale !== 6.0) {
      pipelineNotes.push(`#   embedded_cfg_scale: ${config.embedded_cfg_scale}`)
    }
    if (config.flow_shift !== null) {
      pipelineNotes.push(`#   flow_shift: ${config.flow_shift}`)
    }
    if (config.attention_backend === "VIDEO_SPARSE_ATTN" && config.VSA_sparsity > 0) {
      pipelineNotes.push(`#   VSA_sparsity: ${config.VSA_sparsity}`)
    }

    const lines: string[] = [
      "# Install once:",
      "#   pip install fastvideo",
      "",
      "# Save this file as run.py and run:",
      "#   python run.py",
      "import os",
      "from fastvideo import VideoGenerator, SamplingParam",
      "",
      `os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "${config.attention_backend}"`,
      "",
      "generator = VideoGenerator.from_pretrained(",
      `    "${config.model_id}",`,
      ...engineKwargs,
      ")",
      "",
      `param = SamplingParam.from_pretrained("${config.model_id}")`,
      ...paramAssigns,
      "",
    ]

    if (pipelineNotes.length) {
      lines.push(
        "# Pipeline-config-level overrides (require a custom config file):",
        ...pipelineNotes,
        "",
      )
    }

    lines.push(
      "generator.generate_video(",
      '    "your prompt here",',
      "    sampling_param=param,",
      '    output_path="outputs/",',
      "    save_video=True,",
      ")",
    )

    return lines.join("\n")
  }, [config])

  const copyCommand = () => {
    navigator.clipboard.writeText(command)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const updateConfig = <K extends keyof Config>(key: K, value: Config[K]) => {
    setConfig((prev) => ({ ...prev, [key]: value }))
  }

  const selectedModel = MODEL_BY_ID[config.model_id]
  const isDmdMode = Boolean(selectedModel?.dmdMode)
  const isI2V = config.workload_type === "i2v" || config.workload_type === "ti2v"

  useEffect(() => {
    if (typeof window === "undefined" || window.parent === window) return

    let frame = 0
    const timers: number[] = []

    const sendHeight = () => {
      window.cancelAnimationFrame(frame)
      frame = window.requestAnimationFrame(() => {
        const height = Math.max(
          document.body.scrollHeight,
          document.body.offsetHeight,
          document.documentElement.scrollHeight,
          document.documentElement.offsetHeight,
        )
        window.parent.postMessage({ type: "adv-tuning-guide-height", height }, "*")
      })
    }

    sendHeight()

    const ro = new ResizeObserver(sendHeight)
    ro.observe(document.documentElement)
    ro.observe(document.body)

    const mo = new MutationObserver(sendHeight)
    mo.observe(document.documentElement, {
      attributes: true,
      childList: true,
      characterData: true,
      subtree: true,
    })

    window.addEventListener("load", sendHeight)
    window.addEventListener("resize", sendHeight)

    for (const delay of [100, 300, 1000]) {
      timers.push(window.setTimeout(sendHeight, delay))
    }

    return () => {
      window.cancelAnimationFrame(frame)
      timers.forEach((timer) => window.clearTimeout(timer))
      window.removeEventListener("load", sendHeight)
      window.removeEventListener("resize", sendHeight)
      mo.disconnect()
      ro.disconnect()
    }
  }, [])

  return (
    <div className="min-h-screen bg-white">
    {/* <div className="min-h-screen bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950">*/}
    
      <div className="max-w-6xl">
        {/* Stats Bar */}
        <div className="mb-8 border-b border-blue-200 bg-blue-600 px-8 py-5 shadow-sm">
          <div className="grid grid-cols-2 gap-6 lg:grid-cols-4">
            <div>
              <div className="mb-1 flex items-center gap-2 text-blue-100">
                <Clock className="h-4 w-4" />
                <span className="text-xs font-semibold uppercase tracking-wide">Video Length</span>
                <Tooltip side="bottom" iconClassName="ml-1.5 text-blue-100 transition-colors hover:text-white" content="Playback duration of the output video: num_frames / fps." />
              </div>
              <div className="text-2xl font-semibold text-white">{hasAvailableModelRuns ? `${videoLength}s` : "N/A"}</div>
            </div>
            <div>
              <div className="mb-1 flex items-center gap-2 text-blue-100">
                <Database className="h-4 w-4" />
                <span className="text-xs font-semibold uppercase tracking-wide">Total Pixels</span>
                <Tooltip side="bottom" iconClassName="ml-1.5 text-blue-100 transition-colors hover:text-white" content="Total pixels across all output frames: height * width * num_frames." />
              </div>
              <div className="text-2xl font-semibold text-white">{hasAvailableModelRuns ? `${(totalPixels / 1000000).toFixed(1)}M` : "N/A"}</div>
            </div>
            <div>
              <div className="mb-1 flex items-center gap-2 text-blue-100">
                <BarChart3 className="h-4 w-4" />
                <span className="text-xs font-semibold uppercase tracking-wide">Latent Tokens</span>
                <Tooltip side="bottom" iconClassName="ml-1.5 text-blue-100 transition-colors hover:text-white" content="The sequence length the DiT processes: ((num_frames-1)/4 + 1) * (height/8) * (width/8). Attention cost scales with its square." />
              </div>
              <div className="text-2xl font-semibold text-white">{hasAvailableModelRuns ? `${(latentTokens / 1000).toFixed(0)}K` : "N/A"}</div>
            </div>
            <div>
              <div className="mb-1 flex items-center gap-2 text-blue-100">
                <Gauge className="h-4 w-4" />
                <span className="text-xs font-semibold uppercase tracking-wide">Est. Time</span>
                <Tooltip side="bottom" iconClassName="ml-1.5 text-blue-100 transition-colors hover:text-white" content="Real measured generation time from the FastVideo benchmark for the selected model + GPU (it does not change when you adjust resolution, frames, or steps)." />
              </div>
              {benchmark ? (
                <>
                  <div className="text-2xl font-semibold text-white">
                    {benchmark.run.generationSeconds != null ? `${Math.round(benchmark.run.generationSeconds)}s` : "N/A"}
                  </div>
                  {benchmark.run.generationSeconds != null && (
                    <div className="mt-0.5 text-xs text-blue-100">
                      {`${benchmark.deviated ? "⚠ " : ""}measured @ ${benchmark.run.steps} steps · ${benchmark.run.width}×${benchmark.run.height} · ${ATTENTION_BY_ID[benchmark.run.attention]?.shortName ?? benchmark.run.attention}`}
                    </div>
                  )}
                </>
              ) : !hasAvailableModelRuns ? (
                <>
                  <div className="text-2xl font-semibold text-white">N/A</div>
                </>
              ) : (
                <>
                  <div className="text-2xl font-semibold text-white">—</div>
                  <div className="mt-0.5 text-xs text-blue-100">not benchmarked</div>
                </>
              )}
            </div>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-6 px-2 pb-10">
          {/* Left Column - Parameters */}
          <div className="col-span-2 space-y-4">
            <Section title="Model & Hardware" icon={Monitor} number="A" color="sky">
              <div className="grid grid-cols-2 gap-4 mt-4">
                <div className="col-span-2">
                  <SelectInput
                    label="Model"
                    value={selectedModelRun}
                    onChange={(v) => {
                      const [model, attention] = v.split("::")
                      setConfig((prev) => ({ ...prev, model_id: model, attention_backend: attention }))
                    }}
                    options={
                      hasAvailableModelRuns
                        ? availableModelRuns.map((run) => {
                            const model = MODEL_BY_ID[run.model]
                            const backend = ATTENTION_BY_ID[run.attention]?.shortName ?? run.attention
                            return {
                              id: `${run.model}::${run.attention}`,
                              name: `${model.name} (${model.size}, ${backend})`,
                            }
                          })
                        : [{ id: selectedModelRun, name: "Not suitable" }]
                    }
                    disabled={!hasAvailableModelRuns}
                    tooltip="Only models with a passing benchmark for the selected Workload Type + GPU are listed."
                  />
                </div>
                <SelectInput
                  label="Workload Type"
                  value={config.workload_type}
                  onChange={(v) => updateConfig("workload_type", v)}
                  options={workloadOptions.map(w => ({ id: w.id, name: w.name }))}
                  tooltip="Task category. Determines which pipeline stages run."
                />
                <SelectInput
                  label="GPU"
                  value={config.gpu}
                  onChange={(v) => updateConfig("gpu", v)}
                  options={gpuOptions.map(g => ({ id: g.id, name: `${g.name} (${g.vram})` }))}
                  tooltip="GPU model. Used for benchmark tracking and hardware-aware configuration."
                />
                <NumberInput
                  label="Number of GPUs"
                  value={config.num_gpus}
                  onChange={(v) => updateConfig("num_gpus", v)}
                  min={1}
                  max={8}
                  tooltip="Values > 1 require use_fsdp_inference = true."
                />
              </div>
            </Section>

            <Section title="Video Specification" icon={Film} number="B" color="emerald">
              <div className="mt-4 space-y-4">
                <div>
                  <label className="mb-2 block text-xs font-semibold uppercase tracking-wide text-slate-600">Resolution Preset</label>
                  <div className="grid grid-cols-3 gap-2">
                    {resolutionPresets.map((preset) => (
                      <button
                        key={preset.label}
                        onClick={() => {
                          updateConfig("height", preset.height)
                          updateConfig("width", preset.width)
                        }}
                        className={`rounded-lg border px-3 py-2 text-sm transition-all ${
                          config.height === preset.height && config.width === preset.width
                            ? "border-[var(--section-color)] bg-[var(--section-color)] text-white"
                            : "border-[var(--section-color)] bg-white text-[var(--section-color)] hover:bg-[var(--section-soft)]"
                        }`}
                      >
                        {preset.label}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <NumberInput label="Height" value={config.height} onChange={(v) => updateConfig("height", v)} min={256} max={2160} step={8} unit="px" tooltip="Frame height in pixels. Must be divisible by 8 (VAE requirement)." />
                  <NumberInput label="Width" value={config.width} onChange={(v) => updateConfig("width", v)} min={256} max={3840} step={8} unit="px" tooltip="Frame width in pixels. Must be divisible by 8." />
                </div>
                <div>
                  <label className="mb-2 block text-xs font-semibold uppercase tracking-wide text-slate-600">Frame Count Preset</label>
                  <div className="grid grid-cols-3 gap-2">
                    {framePresets.map((preset) => (
                      <button
                        key={preset.frames}
                        onClick={() => updateConfig("num_frames", preset.frames)}
                        className={`rounded-lg border px-3 py-2 text-sm transition-all ${
                          config.num_frames === preset.frames
                            ? "border-[var(--section-color)] bg-[var(--section-color)] text-white"
                            : "border-[var(--section-color)] bg-white text-[var(--section-color)] hover:bg-[var(--section-soft)]"
                        }`}
                      >
                        {preset.label}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <NumberInput label="Number of Frames" value={config.num_frames} onChange={(v) => updateConfig("num_frames", v)} min={1} max={240} tooltip="Total number of frames to generate." />
                  <NumberInput label="FPS" value={config.fps} onChange={(v) => updateConfig("fps", v)} min={1} max={60} tooltip="Frames per second for the output video. Does NOT affect generation time." />
                </div>
              </div>
            </Section>

            <Section title="Quality Controls" icon={Sparkles} number="C" color="amber">
              <div className="mt-4 space-y-4">
                <div>
                  <label className="mb-2 flex items-center text-xs font-semibold uppercase tracking-wide text-slate-600">
                    Inference Steps
                    <Tooltip content="Number of denoising iterations. Time scales linearly." />
                  </label>
                  <div className="flex items-center gap-4">
                    <input type="range" min={10} max={80} value={config.num_inference_steps} onChange={(e) => updateConfig("num_inference_steps", Number(e.target.value))} className="h-2 flex-1 cursor-pointer appearance-none rounded-full bg-[var(--section-soft)] accent-[var(--section-color)] [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-[var(--section-color)]" />
                    <span className="w-12 text-center font-mono text-sm text-[var(--section-color)]">{config.num_inference_steps}</span>
                  </div>
                  <div className="mt-1 flex justify-between text-xs text-slate-500">
                    <span>Draft (10)</span><span>Balanced (30)</span><span>High (80)</span>
                  </div>
                </div>
                <div>
                  <label className="mb-2 flex items-center text-xs font-semibold uppercase tracking-wide text-slate-600">
                    Guidance Scale
                    <Tooltip content="Controls how strongly the model follows the text prompt. No time cost." />
                  </label>
                  <div className="flex items-center gap-4">
                    <input type="range" min={1} max={10} step={0.5} value={config.guidance_scale} onChange={(e) => updateConfig("guidance_scale", Number(e.target.value))} className="h-2 flex-1 cursor-pointer appearance-none rounded-full bg-[var(--section-soft)] accent-[var(--section-color)] [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-[var(--section-color)]" />
                    <span className="w-12 text-center font-mono text-sm text-[var(--section-color)]">{config.guidance_scale}</span>
                  </div>
                  <div className="mt-1 flex justify-between text-xs text-slate-500">
                    <span>Loose (3)</span><span>Balanced (5)</span><span>Strict (8+)</span>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <NumberInput label="Embedded CFG Scale" value={config.embedded_cfg_scale} onChange={(v) => updateConfig("embedded_cfg_scale", v)} min={1} max={10} step={0.5} tooltip="Model-level CFG signal passed into the DiT. Default: 6.0. Rarely change." />
                  <NumberInput label="Seed" value={config.seed} onChange={(v) => updateConfig("seed", v)} min={-1} tooltip="Random seed. Set to -1 for random. Fix a seed for reproducibility." />
                </div>
              </div>
            </Section>

            <Section title="Speed Optimization" icon={Zap} number="D" color="violet">
              <div className="mt-4 space-y-4">
                <div>
                  <label className="mb-2 block text-xs font-semibold uppercase tracking-wide text-slate-600">Attention Backend</label>
                  <div className="grid grid-cols-2 gap-2">
                    {attentionBackends.map((backend) => (
                      <button
                        key={backend.id}
                        onClick={() => updateConfig("attention_backend", backend.id)}
                        className={`rounded-lg border p-3 text-left transition-all ${
                          config.attention_backend === backend.id
                            ? "border-[var(--section-color)] bg-[var(--section-soft)]"
                            : "border-slate-200 bg-white hover:border-[var(--section-color)]"
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <span className={`text-sm font-medium ${config.attention_backend === backend.id ? "text-slate-900" : "text-slate-900"}`}>{backend.name}</span>
                          <span className={`rounded-full px-2 py-0.5 text-xs ${backend.speed === "Baseline" ? "border border-slate-400 bg-white text-slate-600" : "border border-[var(--section-color)] bg-white text-[var(--section-color)]"}`}>{backend.speed}</span>
                        </div>
                        <div className="mt-1 text-xs text-slate-500">{backend.desc}</div>
                      </button>
                    ))}
                  </div>
                </div>
                {config.attention_backend === "VIDEO_SPARSE_ATTN" && (
                  <div>
                    <label className="mb-2 flex items-center text-xs font-semibold uppercase tracking-wide text-slate-600">
                      VSA Sparsity
                      <Tooltip content="Higher sparsity = faster but with some quality reduction. 0 = dense (no quality loss)." />
                    </label>
                    <div className="flex items-center gap-4">
                      <input type="range" min={0} max={1} step={0.1} value={config.VSA_sparsity} onChange={(e) => updateConfig("VSA_sparsity", Number(e.target.value))} className="h-2 flex-1 cursor-pointer appearance-none rounded-full bg-[var(--section-soft)] accent-[var(--section-color)] [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-[var(--section-color)]" />
                      <span className="w-12 text-center font-mono text-sm text-[var(--section-color)]">{config.VSA_sparsity}</span>
                    </div>
                  </div>
                )}
                {isDmdMode && (
                  <div className="rounded-lg border border-[var(--section-border)] bg-[var(--section-soft)] p-3">
                    <div className="mb-1 flex items-center gap-2 text-sm font-medium text-[var(--section-color)]">
                      <Zap className="h-4 w-4" />
                      DMD Mode Available
                    </div>
                    <div className="text-xs text-slate-600">{selectedModel?.dmdNote}</div>
                    <div className="mt-2 rounded bg-[var(--section-color)] px-2 py-1 font-mono text-xs text-white">dmd_steps: {config.dmd_denoising_steps}</div>
                  </div>
                )}
              </div>
            </Section>

            <Section title="Memory Management" icon={HardDrive} number="E" color="rose">
              <div className="mt-4 space-y-3">
                <div className="mb-4 rounded-lg border border-[var(--section-border)] bg-[var(--section-soft)] p-3">
                  <div className="text-xs text-slate-600">
                    <strong className="text-[var(--section-color)]">Tip:</strong> For 24GB VRAM with 14B models, enable dit_cpu_offload + vae_cpu_offload + pin_cpu_memory.
                  </div>
                </div>
                <Toggle label="dit_cpu_offload" checked={config.dit_cpu_offload} onChange={(v) => updateConfig("dit_cpu_offload", v)} tooltip="Offload entire DiT to RAM. Saves ~4GB, ~2-3x slower." />
                <Toggle label="dit_layerwise_offload" checked={config.dit_layerwise_offload} onChange={(v) => updateConfig("dit_layerwise_offload", v)} tooltip="Layer-by-layer offload. Maximum VRAM saving (~8GB), ~4x slower." />
                <Toggle label="vae_cpu_offload" checked={config.vae_cpu_offload} onChange={(v) => updateConfig("vae_cpu_offload", v)} tooltip="Offload VAE decoder. Saves ~2GB. Minimal time penalty." />
                <Toggle label="text_encoder_cpu_offload" checked={config.text_encoder_cpu_offload} onChange={(v) => updateConfig("text_encoder_cpu_offload", v)} tooltip="Offload text encoder. Saves ~1-2GB. Runs once at start." />
                <Toggle label="image_encoder_cpu_offload" checked={config.image_encoder_cpu_offload} onChange={(v) => updateConfig("image_encoder_cpu_offload", v)} tooltip="I2V/TI2V only. Saves ~1-2GB." disabled={!isI2V} />
                <Toggle label="pin_cpu_memory" checked={config.pin_cpu_memory} onChange={(v) => updateConfig("pin_cpu_memory", v)} tooltip="Pins RAM pages for faster PCIe DMA transfers. Always enable with any offload flag." />
                <Toggle label="use_fsdp_inference" checked={config.use_fsdp_inference} onChange={(v) => updateConfig("use_fsdp_inference", v)} tooltip="Shards model weights across multiple GPUs via FSDP." disabled={config.num_gpus <= 1} />
              </div>
            </Section>
          </div>

          {/* Right Column - Output */}
          <div className="space-y-4">
            <div>
              <div className="overflow-hidden rounded-xl border border-blue-200 bg-white shadow-sm">
                <div className="flex items-center justify-between border-b border-blue-100 bg-blue-50 p-4">
                  <div className="flex items-center gap-2">
                    <span className="flex h-6 w-6 items-center justify-center rounded-full bg-blue-600 text-xs font-medium text-white">
                      <Check className="h-3.5 w-3.5" />
                    </span>
                    <h2 className="text-sm font-semibold uppercase tracking-wider text-blue-600">Generated Command</h2>
                  </div>
                  {hasAvailableModelRuns && (
                    <button onClick={copyCommand} className="flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium text-slate-600 transition-all hover:bg-blue-100 hover:text-blue-700">
                      {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                      {copied ? "Copied" : "Copy"}
                    </button>
                  )}
                </div>
                <div className="max-h-[60vh] overflow-y-auto bg-blue-950 p-4">
                  {hasAvailableModelRuns ? (
                    <pre className="whitespace-pre-wrap font-mono text-sm leading-relaxed text-blue-50">{command}</pre>
                  ) : (
                    <div className="rounded-lg border border-dashed border-blue-700 bg-blue-900/40 px-4 py-6 text-sm leading-relaxed text-blue-100">
                      <div className="font-semibold text-white">Not suitable</div>
                    </div>
                  )}
                </div>
              </div>

              <div className="mt-4 rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
                <h3 className="mb-3 text-xs font-semibold uppercase tracking-wide text-blue-600">Quick Reference</h3>
                <div className="space-y-2 text-xs">
                  <div className="flex justify-between text-slate-600"><span>Resolution</span><span className="font-mono text-slate-900">{hasAvailableModelRuns ? `${config.width}x${config.height}` : "N/A"}</span></div>
                  <div className="flex justify-between text-slate-600"><span>Frames</span><span className="font-mono text-slate-900">{hasAvailableModelRuns ? config.num_frames : "N/A"}</span></div>
                  <div className="flex justify-between text-slate-600"><span>Steps</span><span className="font-mono text-slate-900">{hasAvailableModelRuns ? config.num_inference_steps : "N/A"}</span></div>
                  <div className="flex justify-between text-slate-600"><span>Backend</span><span className="font-mono text-slate-900">{hasAvailableModelRuns ? config.attention_backend : "N/A"}</span></div>
                  <div className="flex justify-between text-slate-600">
                    <span>Offloading</span>
                    <span className="font-mono text-slate-900">{hasAvailableModelRuns ? ([config.dit_cpu_offload && "DiT", config.vae_cpu_offload && "VAE"].filter(Boolean).join("+") || "None") : "N/A"}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
