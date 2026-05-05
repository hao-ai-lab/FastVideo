"use client"

import { useState, useMemo } from "react"
import { 
  Monitor, Cpu, Film, Sparkles, Zap, HardDrive, BarChart3, 
  Copy, Check, Info, ChevronDown, ChevronUp, Settings2,
  Layers, Clock, Database, Gauge
} from "lucide-react"
import Link from "next/link"

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

const modelOptions = [
  { id: "FastVideo/FastWan2.1-T2V-1.3B-Diffusers", name: "FastWan 1.3B (T2V)", size: "1.3B" },
  { id: "Wan-AI/Wan2.1-T2V-14B-Diffusers", name: "Wan 14B (T2V)", size: "14B" },
  { id: "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", name: "Wan 14B I2V 480P", size: "14B" },
  { id: "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers", name: "Wan 14B I2V 720P", size: "14B" },
]

const workloadOptions = [
  { id: "t2v", name: "Text to Video", icon: Film },
  { id: "i2v", name: "Image to Video", icon: Sparkles },
  { id: "ti2v", name: "Text+Image to Video", icon: Layers },
  { id: "action", name: "Action", icon: Zap },
]

const gpuOptions = [
  { id: "rtx4090", name: "RTX 4090", vram: "24GB" },
  { id: "rtx5090", name: "RTX 5090", vram: "32GB" },
  { id: "a100", name: "A100", vram: "40/80GB" },
  { id: "h100", name: "H100", vram: "80GB" },
]

const resolutionPresets = [
  { height: 480, width: 832, label: "480p (832×480)" },
  { height: 720, width: 1280, label: "720p (1280×720)" },
  { height: 1080, width: 1920, label: "1080p (1920×1080)" },
]

const framePresets = [
  { frames: 49, label: "49 frames (~2s @24fps)" },
  { frames: 81, label: "81 frames (~3.4s @24fps)" },
  { frames: 121, label: "121 frames (~5s @24fps)" },
]

const attentionBackends = [
  { id: "FLASH_ATTN", name: "Flash Attention", desc: "Best quality-speed trade-off", speed: "Baseline" },
  { id: "VIDEO_SPARSE_ATTN", name: "Video Sparse Attention", desc: "~30% faster, minimal quality loss", speed: "Fast" },
  { id: "SAGE_ATTN", name: "Sage Attention", desc: "~40% faster, visible at high motion", speed: "Faster" },
  { id: "SLA_ATTN", name: "Sliding Tile Attention", desc: "Check model compatibility", speed: "Fast" },
]

// Tooltip component
function Tooltip({ content }: { content: string }) {
  const [show, setShow] = useState(false)
  return (
    <div className="relative inline-block">
      <button
        onMouseEnter={() => setShow(true)}
        onMouseLeave={() => setShow(false)}
        className="ml-1.5 text-slate-500 hover:text-sky-400 transition-colors"
      >
        <Info className="h-3.5 w-3.5" />
      </button>
      {show && (
        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-64 rounded-lg bg-slate-800 border border-white/10 p-3 text-xs text-slate-300 shadow-xl z-50">
          {content}
          <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-slate-800" />
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
  const colorClasses = {
    sky: "bg-sky-500/20 text-sky-400",
    emerald: "bg-emerald-500/20 text-emerald-400",
    amber: "bg-amber-500/20 text-amber-400",
    violet: "bg-violet-500/20 text-violet-400",
    rose: "bg-rose-500/20 text-rose-400",
    slate: "bg-slate-500/20 text-slate-400",
  }
  const textColors = {
    sky: "text-sky-400",
    emerald: "text-emerald-400",
    amber: "text-amber-400",
    violet: "text-violet-400",
    rose: "text-rose-400",
    slate: "text-slate-400",
  }
  
  return (
    <div className="rounded-xl border border-white/5 bg-white/[0.02] overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between p-4 hover:bg-white/[0.02] transition-colors"
      >
        <div className="flex items-center gap-3">
          <span className={`flex h-8 w-8 items-center justify-center rounded-lg ${colorClasses[color]} text-sm font-semibold`}>
            {number}
          </span>
          <Icon className={`h-5 w-5 ${textColors[color]}`} />
          <h2 className={`text-sm font-semibold uppercase tracking-wider ${textColors[color]}`}>{title}</h2>
        </div>
        {open ? <ChevronUp className="h-5 w-5 text-slate-500" /> : <ChevronDown className="h-5 w-5 text-slate-500" />}
      </button>
      {open && <div className="p-4 pt-0 border-t border-white/5">{children}</div>}
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
      <label className="flex items-center text-xs font-semibold uppercase tracking-wide text-slate-300 mb-2">
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
          className="w-full rounded-lg border border-white/10 bg-slate-900/50 px-3 py-2.5 text-sm text-white focus:border-sky-500/50 focus:outline-none focus:ring-1 focus:ring-sky-500/50"
        />
        {unit && <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-slate-500">{unit}</span>}
      </div>
    </div>
  )
}

function SelectInput({ 
  label, 
  value, 
  onChange, 
  options,
  tooltip
}: { 
  label: string
  value: string
  onChange: (v: string) => void
  options: { id: string; name: string; desc?: string }[]
  tooltip?: string
}) {
  return (
    <div>
      <label className="flex items-center text-xs font-semibold uppercase tracking-wide text-slate-300 mb-2">
        {label}
        {tooltip && <Tooltip content={tooltip} />}
      </label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full rounded-lg border border-white/10 bg-slate-900/50 px-3 py-2.5 text-sm text-white focus:border-sky-500/50 focus:outline-none focus:ring-1 focus:ring-sky-500/50"
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
    <label className={`flex items-center justify-between p-3 rounded-lg border border-white/5 bg-white/[0.02] cursor-pointer hover:bg-white/[0.04] transition-colors ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}>
      <div className="flex items-center gap-2">
        <span className="text-sm text-slate-300">{label}</span>
        {tooltip && <Tooltip content={tooltip} />}
      </div>
      <button
        type="button"
        disabled={disabled}
        onClick={() => !disabled && onChange(!checked)}
        className={`relative h-6 w-11 rounded-full transition-colors ${checked ? 'bg-sky-500' : 'bg-slate-700'}`}
      >
        <span className={`absolute top-0.5 left-0.5 h-5 w-5 rounded-full bg-white shadow transition-transform ${checked ? 'translate-x-5' : ''}`} />
      </button>
    </label>
  )
}

export default function AdvancedTuningPage() {
  const [copied, setCopied] = useState(false)
  const [config, setConfig] = useState<Config>({
    // Category A
    model_id: "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
    workload_type: "t2v",
    gpu: "rtx4090",
    num_gpus: 1,
    // Category B
    height: 480,
    width: 832,
    num_frames: 49,
    fps: 24,
    // Category C
    num_inference_steps: 30,
    guidance_scale: 5.0,
    embedded_cfg_scale: 6.0,
    flow_shift: null,
    seed: -1,
    // Category D
    attention_backend: "FLASH_ATTN",
    VSA_sparsity: 0,
    dmd_denoising_steps: "1000,757,522",
    // Category E
    dit_cpu_offload: false,
    dit_layerwise_offload: false,
    vae_cpu_offload: false,
    text_encoder_cpu_offload: false,
    image_encoder_cpu_offload: false,
    pin_cpu_memory: false,
    use_fsdp_inference: false,
  })

  // Derived values
  const videoLength = useMemo(() => (config.num_frames / config.fps).toFixed(1), [config.num_frames, config.fps])
  const totalPixels = useMemo(() => config.height * config.width * config.num_frames, [config.height, config.width, config.num_frames])
  const computeCost = useMemo(() => totalPixels * config.num_inference_steps, [totalPixels, config.num_inference_steps])

  // Estimate generation time (rough approximation)
  const estimatedTime = useMemo(() => {
    const baseTime = (totalPixels / 19600000) * 60 * (config.num_inference_steps / 30)
    let multiplier = 1
    if (config.dit_cpu_offload) multiplier *= 2.5
    if (config.dit_layerwise_offload) multiplier *= 4
    if (config.attention_backend === "VIDEO_SPARSE_ATTN") multiplier *= 0.7
    if (config.attention_backend === "SAGE_ATTN") multiplier *= 0.6
    return Math.round(baseTime * multiplier)
  }, [totalPixels, config.num_inference_steps, config.dit_cpu_offload, config.dit_layerwise_offload, config.attention_backend])

  // Generate command
  const command = useMemo(() => {
    const lines = [
      "# 1. Install FastVideo",
      "pip install fastvideo",
      "",
      "# 2. Download model",
      `huggingface-cli download ${config.model_id} \\`,
      `  --local-dir ./models/${config.model_id.split('/').pop()}`,
      "",
      "# 3. Run generation",
      "fastvideo generate \\",
      `  --model ./models/${config.model_id.split('/').pop()} \\`,
      `  --attention ${config.attention_backend} \\`,
      `  --height ${config.height} --width ${config.width} \\`,
      `  --num_frames ${config.num_frames} \\`,
      `  --num_inference_steps ${config.num_inference_steps} \\`,
      `  --guidance_scale ${config.guidance_scale} \\`,
    ]

    if (config.embedded_cfg_scale !== 6.0) {
      lines.push(`  --embedded_cfg_scale ${config.embedded_cfg_scale} \\`)
    }
    if (config.flow_shift !== null) {
      lines.push(`  --flow_shift ${config.flow_shift} \\`)
    }
    if (config.seed !== -1) {
      lines.push(`  --seed ${config.seed} \\`)
    }
    if (config.attention_backend === "VIDEO_SPARSE_ATTN" && config.VSA_sparsity > 0) {
      lines.push(`  --vsa_sparsity ${config.VSA_sparsity} \\`)
    }
    if (config.dit_cpu_offload) lines.push("  --dit_cpu_offload \\")
    if (config.dit_layerwise_offload) lines.push("  --dit_layerwise_offload \\")
    if (config.vae_cpu_offload) lines.push("  --vae_cpu_offload \\")
    if (config.text_encoder_cpu_offload) lines.push("  --text_encoder_cpu_offload \\")
    if (config.image_encoder_cpu_offload && (config.workload_type === "i2v" || config.workload_type === "ti2v")) {
      lines.push("  --image_encoder_cpu_offload \\")
    }
    if (config.pin_cpu_memory) lines.push("  --pin_cpu_memory \\")
    if (config.use_fsdp_inference && config.num_gpus > 1) lines.push("  --use_fsdp_inference \\")
    
    if (config.workload_type === "i2v" || config.workload_type === "ti2v") {
      lines.push("  --image ./input.png \\")
    }
    lines.push('  --prompt "your prompt here"')

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

  const isFastWanDMD = config.model_id.includes("FastWan")
  const isI2V = config.workload_type === "i2v" || config.workload_type === "ti2v"

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950">
      {/* Header */}
      <div className="border-b border-white/5 bg-slate-900/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="mx-auto max-w-5xl px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link href="/">
                <img 
                  src="https://hebbkx1anhila5yf.public.blob.vercel-storage.com/image-xVE4qNY0zoz6VgwQkgqPAbcJG4rfbP.png" 
                  alt="FastVideo Logo" 
                  className="h-10 w-auto"
                />
              </Link>
              <div>
                <h1 className="text-lg font-semibold text-white">Advanced Tuning Guide</h1>
                <p className="text-sm text-slate-300">RTX 4090 / RTX 5090 - All 27 Parameters</p>
              </div>
            </div>
            <Link 
              href="/"
              className="text-sm text-slate-400 hover:text-sky-400 transition-colors"
            >
              Quick Start
            </Link>
          </div>
        </div>
      </div>

      <div className="mx-auto max-w-5xl px-6 py-8">
        {/* Stats Bar */}
        <div className="sticky top-[73px] z-40 -mx-6 mb-8 border-b border-white/5 bg-slate-950/90 px-6 py-3 backdrop-blur">
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <div className="rounded-xl border border-white/5 bg-white/[0.02] p-4">
              <div className="flex items-center gap-2 text-sky-400 mb-1">
                <Clock className="h-4 w-4" />
                <span className="text-xs font-semibold uppercase tracking-wide">Video Length</span>
              </div>
              <div className="text-2xl font-semibold text-white">{videoLength}s</div>
            </div>
            <div className="rounded-xl border border-white/5 bg-white/[0.02] p-4">
              <div className="flex items-center gap-2 text-emerald-400 mb-1">
                <Gauge className="h-4 w-4" />
                <span className="text-xs font-semibold uppercase tracking-wide">Est. Time</span>
              </div>
              <div className="text-2xl font-semibold text-white">~{estimatedTime}s</div>
            </div>
            <div className="rounded-xl border border-white/5 bg-white/[0.02] p-4">
              <div className="flex items-center gap-2 text-amber-400 mb-1">
                <Database className="h-4 w-4" />
                <span className="text-xs font-semibold uppercase tracking-wide">Total Pixels</span>
              </div>
              <div className="text-2xl font-semibold text-white">{(totalPixels / 1000000).toFixed(1)}M</div>
            </div>
            <div className="rounded-xl border border-white/5 bg-white/[0.02] p-4">
              <div className="flex items-center gap-2 text-violet-400 mb-1">
                <BarChart3 className="h-4 w-4" />
                <span className="text-xs font-semibold uppercase tracking-wide">Compute Cost</span>
              </div>
              <div className="text-2xl font-semibold text-white">{(computeCost / 1000000000).toFixed(2)}G</div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-6">
          {/* Left Column - Parameters */}
          <div className="col-span-2 space-y-4">
            {/* Category A - Model & Hardware */}
            <Section title="Model & Hardware" icon={Monitor} number="A" color="sky">
              <div className="grid grid-cols-2 gap-4 mt-4">
                <div className="col-span-2">
                  <SelectInput
                    label="Model"
                    value={config.model_id}
                    onChange={(v) => updateConfig("model_id", v)}
                    options={modelOptions.map(m => ({ id: m.id, name: `${m.name} (${m.size})` }))}
                    tooltip="HuggingFace model path - uniquely identifies the model weights and architecture."
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

            {/* Category B - Video Spec */}
            <Section title="Video Specification" icon={Film} number="B" color="emerald">
              <div className="mt-4 space-y-4">
                <div>
                  <label className="text-xs font-semibold uppercase tracking-wide text-slate-300 mb-2 block">Resolution Preset</label>
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
                            ? "border-emerald-500/50 bg-emerald-500/10 text-emerald-400"
                            : "border-white/10 bg-white/[0.02] text-slate-400 hover:border-white/20"
                        }`}
                      >
                        {preset.label}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <NumberInput
                    label="Height"
                    value={config.height}
                    onChange={(v) => updateConfig("height", v)}
                    min={256}
                    max={2160}
                    step={8}
                    unit="px"
                    tooltip="Frame height in pixels. Must be divisible by 8 (VAE requirement)."
                  />
                  <NumberInput
                    label="Width"
                    value={config.width}
                    onChange={(v) => updateConfig("width", v)}
                    min={256}
                    max={3840}
                    step={8}
                    unit="px"
                    tooltip="Frame width in pixels. Must be divisible by 8."
                  />
                </div>
                <div>
                  <label className="text-xs font-semibold uppercase tracking-wide text-slate-300 mb-2 block">Frame Count Preset</label>
                  <div className="grid grid-cols-3 gap-2">
                    {framePresets.map((preset) => (
                      <button
                        key={preset.frames}
                        onClick={() => updateConfig("num_frames", preset.frames)}
                        className={`rounded-lg border px-3 py-2 text-sm transition-all ${
                          config.num_frames === preset.frames
                            ? "border-emerald-500/50 bg-emerald-500/10 text-emerald-400"
                            : "border-white/10 bg-white/[0.02] text-slate-400 hover:border-white/20"
                        }`}
                      >
                        {preset.label}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <NumberInput
                    label="Number of Frames"
                    value={config.num_frames}
                    onChange={(v) => updateConfig("num_frames", v)}
                    min={1}
                    max={240}
                    tooltip="Total number of frames to generate. Directly sets video duration at a given fps."
                  />
                  <NumberInput
                    label="FPS"
                    value={config.fps}
                    onChange={(v) => updateConfig("fps", v)}
                    min={1}
                    max={60}
                    tooltip="Frames per second for the output video. Does NOT affect generation time."
                  />
                </div>
              </div>
            </Section>

            {/* Category C - Quality */}
            <Section title="Quality Controls" icon={Sparkles} number="C" color="amber">
              <div className="mt-4 space-y-4">
                <div>
                  <label className="flex items-center text-xs font-semibold uppercase tracking-wide text-slate-300 mb-2">
                    Inference Steps
                    <Tooltip content="Number of denoising iterations. Time scales linearly." />
                  </label>
                  <div className="flex items-center gap-4">
                    <input
                      type="range"
                      min={10}
                      max={80}
                      value={config.num_inference_steps}
                      onChange={(e) => updateConfig("num_inference_steps", Number(e.target.value))}
                      className="flex-1 h-2 rounded-full bg-slate-700 appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-amber-400"
                    />
                    <span className="w-12 text-center text-sm font-mono text-white">{config.num_inference_steps}</span>
                  </div>
                  <div className="flex justify-between text-xs text-slate-500 mt-1">
                    <span>Draft (10)</span>
                    <span>Balanced (30)</span>
                    <span>High (80)</span>
                  </div>
                </div>
                <div>
                  <label className="flex items-center text-xs font-semibold uppercase tracking-wide text-slate-300 mb-2">
                    Guidance Scale
                    <Tooltip content="Controls how strongly the model follows the text prompt. No time cost." />
                  </label>
                  <div className="flex items-center gap-4">
                    <input
                      type="range"
                      min={1}
                      max={10}
                      step={0.5}
                      value={config.guidance_scale}
                      onChange={(e) => updateConfig("guidance_scale", Number(e.target.value))}
                      className="flex-1 h-2 rounded-full bg-slate-700 appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-amber-400"
                    />
                    <span className="w-12 text-center text-sm font-mono text-white">{config.guidance_scale}</span>
                  </div>
                  <div className="flex justify-between text-xs text-slate-500 mt-1">
                    <span>Loose (3)</span>
                    <span>Balanced (5)</span>
                    <span>Strict (8+)</span>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <NumberInput
                    label="Embedded CFG Scale"
                    value={config.embedded_cfg_scale}
                    onChange={(v) => updateConfig("embedded_cfg_scale", v)}
                    min={1}
                    max={10}
                    step={0.5}
                    tooltip="Model-level CFG signal passed into the DiT. Default: 6.0. Rarely change."
                  />
                  <NumberInput
                    label="Seed"
                    value={config.seed}
                    onChange={(v) => updateConfig("seed", v)}
                    min={-1}
                    tooltip="Random seed. Set to -1 for random. Fix a seed for reproducibility."
                  />
                </div>
              </div>
            </Section>

            {/* Category D - Speed */}
            <Section title="Speed Optimization" icon={Zap} number="D" color="violet">
              <div className="mt-4 space-y-4">
                <div>
                  <label className="text-xs font-semibold uppercase tracking-wide text-slate-300 mb-2 block">
                    Attention Backend
                  </label>
                  <div className="grid grid-cols-2 gap-2">
                    {attentionBackends.map((backend) => (
                      <button
                        key={backend.id}
                        onClick={() => updateConfig("attention_backend", backend.id)}
                        className={`rounded-lg border p-3 text-left transition-all ${
                          config.attention_backend === backend.id
                            ? "border-violet-500/50 bg-violet-500/10"
                            : "border-white/10 bg-white/[0.02] hover:border-white/20"
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <span className={`text-sm font-medium ${config.attention_backend === backend.id ? "text-violet-400" : "text-white"}`}>
                            {backend.name}
                          </span>
                          <span className={`text-xs px-2 py-0.5 rounded-full ${
                            backend.speed === "Baseline" ? "bg-slate-700 text-slate-300" :
                            backend.speed === "Fast" ? "bg-emerald-500/20 text-emerald-400" :
                            "bg-amber-500/20 text-amber-400"
                          }`}>
                            {backend.speed}
                          </span>
                        </div>
                        <div className="text-xs text-slate-500 mt-1">{backend.desc}</div>
                      </button>
                    ))}
                  </div>
                </div>
                {config.attention_backend === "VIDEO_SPARSE_ATTN" && (
                  <div>
                    <label className="flex items-center text-xs font-semibold uppercase tracking-wide text-slate-300 mb-2">
                      VSA Sparsity
                      <Tooltip content="Higher sparsity = faster but with some quality reduction. 0 = dense (no quality loss)." />
                    </label>
                    <div className="flex items-center gap-4">
                      <input
                        type="range"
                        min={0}
                        max={1}
                        step={0.1}
                        value={config.VSA_sparsity}
                        onChange={(e) => updateConfig("VSA_sparsity", Number(e.target.value))}
                        className="flex-1 h-2 rounded-full bg-slate-700 appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-violet-400"
                      />
                      <span className="w-12 text-center text-sm font-mono text-white">{config.VSA_sparsity}</span>
                    </div>
                  </div>
                )}
                {isFastWanDMD && config.gpu === "rtx4090" && (
                  <div className="rounded-lg border border-violet-500/20 bg-violet-500/5 p-3">
                    <div className="flex items-center gap-2 text-violet-400 text-sm font-medium mb-1">
                      <Zap className="h-4 w-4" />
                      DMD Mode Available
                    </div>
                    <div className="text-xs text-slate-400">
                      FastWan DMD uses 3-step distillation for ~10x faster generation.
                    </div>
                    <div className="mt-2 font-mono text-xs text-slate-300 bg-slate-900/50 rounded px-2 py-1">
                      dmd_steps: {config.dmd_denoising_steps}
                    </div>
                  </div>
                )}
              </div>
            </Section>

            {/* Category E - Memory */}
            <Section title="Memory Management" icon={HardDrive} number="E" color="rose">
              <div className="mt-4 space-y-3">
                <div className="rounded-lg border border-rose-500/20 bg-rose-500/5 p-3 mb-4">
                  <div className="text-xs text-slate-300">
                    <strong className="text-rose-400">Tip:</strong> For 24GB VRAM with 14B models, enable dit_cpu_offload + vae_cpu_offload + pin_cpu_memory.
                  </div>
                </div>
                <Toggle
                  label="dit_cpu_offload"
                  checked={config.dit_cpu_offload}
                  onChange={(v) => updateConfig("dit_cpu_offload", v)}
                  tooltip="Offload entire DiT to RAM. Saves ~4GB, ~2-3x slower."
                />
                <Toggle
                  label="dit_layerwise_offload"
                  checked={config.dit_layerwise_offload}
                  onChange={(v) => updateConfig("dit_layerwise_offload", v)}
                  tooltip="Layer-by-layer offload. Maximum VRAM saving (~8GB), ~4x slower."
                />
                <Toggle
                  label="vae_cpu_offload"
                  checked={config.vae_cpu_offload}
                  onChange={(v) => updateConfig("vae_cpu_offload", v)}
                  tooltip="Offload VAE decoder. Saves ~2GB. Minimal time penalty."
                />
                <Toggle
                  label="text_encoder_cpu_offload"
                  checked={config.text_encoder_cpu_offload}
                  onChange={(v) => updateConfig("text_encoder_cpu_offload", v)}
                  tooltip="Offload text encoder. Saves ~1-2GB. Runs once at start."
                />
                <Toggle
                  label="image_encoder_cpu_offload"
                  checked={config.image_encoder_cpu_offload}
                  onChange={(v) => updateConfig("image_encoder_cpu_offload", v)}
                  tooltip="I2V/TI2V only. Saves ~1-2GB."
                  disabled={!isI2V}
                />
                <Toggle
                  label="pin_cpu_memory"
                  checked={config.pin_cpu_memory}
                  onChange={(v) => updateConfig("pin_cpu_memory", v)}
                  tooltip="Pins RAM pages for faster PCIe DMA transfers. Always enable with any offload flag."
                />
                <Toggle
                  label="use_fsdp_inference"
                  checked={config.use_fsdp_inference}
                  onChange={(v) => updateConfig("use_fsdp_inference", v)}
                  tooltip="Shards model weights across multiple GPUs via FSDP."
                  disabled={config.num_gpus <= 1}
                />
              </div>
            </Section>
          </div>

          {/* Right Column - Output */}
          <div className="space-y-4">
            <div className="sticky top-24">
              {/* Generated Command */}
              <div className="rounded-xl border border-white/5 bg-slate-900/80 overflow-hidden">
                <div className="flex items-center justify-between p-4 border-b border-white/5">
                  <div className="flex items-center gap-2">
                    <span className="flex h-6 w-6 items-center justify-center rounded-full bg-emerald-500/20 text-xs font-medium text-emerald-400">
                      <Check className="h-3.5 w-3.5" />
                    </span>
                    <h2 className="text-sm font-semibold uppercase tracking-wider text-emerald-400">Generated Command</h2>
                  </div>
                  <button
                    onClick={copyCommand}
                    className="flex items-center gap-2 rounded-lg border border-white/10 bg-white/5 px-4 py-2 text-sm font-medium text-slate-300 transition-all hover:border-sky-500/30 hover:bg-sky-500/10 hover:text-sky-400"
                  >
                    {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                    {copied ? "Copied" : "Copy"}
                  </button>
                </div>
                <div className="p-4 max-h-[60vh] overflow-y-auto">
                  <pre className="font-mono text-sm leading-relaxed text-slate-300 whitespace-pre-wrap">
                    {command}
                  </pre>
                </div>
              </div>

              {/* Quick Reference */}
              <div className="mt-4 rounded-xl border border-white/5 bg-white/[0.02] p-4">
                <h3 className="text-xs font-semibold uppercase tracking-wide text-sky-400 mb-3">Quick Reference</h3>
                <div className="space-y-2 text-xs">
                  <div className="flex justify-between text-slate-400">
                    <span>Resolution</span>
                    <span className="text-white font-mono">{config.width}x{config.height}</span>
                  </div>
                  <div className="flex justify-between text-slate-400">
                    <span>Frames</span>
                    <span className="text-white font-mono">{config.num_frames}</span>
                  </div>
                  <div className="flex justify-between text-slate-400">
                    <span>Steps</span>
                    <span className="text-white font-mono">{config.num_inference_steps}</span>
                  </div>
                  <div className="flex justify-between text-slate-400">
                    <span>Backend</span>
                    <span className="text-white font-mono">{config.attention_backend}</span>
                  </div>
                  <div className="flex justify-between text-slate-400">
                    <span>Offloading</span>
                    <span className="text-white font-mono">
                      {[config.dit_cpu_offload && "DiT", config.vae_cpu_offload && "VAE"].filter(Boolean).join("+") || "None"}
                    </span>
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
