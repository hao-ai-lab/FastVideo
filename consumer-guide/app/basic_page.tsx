"use client"

import { useEffect, useState, useMemo } from "react"
import { Monitor, Cpu, HardDrive, Layers, Info, Copy, Check, Film, Image, Sparkles, Gamepad2 } from "lucide-react"
import quickstartData from "@/data/quickstart.json"

type IconName = "Film" | "Image" | "Sparkles" | "Gamepad2"

interface Config {
  model: string
  modelShort: string
  attn: string
  h: number
  w: number
  frames: number
  steps: number
  cfg: number
  vramMin: number
  ramMin: number
  timeSec: number
  firstRunSec?: number
  vram: string
  offload: boolean
  ditCpuOffload?: boolean
  ditLayerwiseOffload?: boolean
  vaeCpuOffload?: boolean
  textEncoderCpuOffload?: boolean
  pinCpuMemory?: boolean
  useFsdpInference?: boolean
  fps?: number
  keyboardDim?: number
  extra?: string
}

interface TaskOption {
  id: string
  label: string
  sub: string
  icon: string
}

interface HardwareOption {
  id: string
  label: string
  defaultVramGb: number
  vramOptionsGb: number[]
}

interface TierOption {
  id: string
  name: string
  desc: string
  badge?: string
}

interface Profile extends Config {
  gpu: string
  vramGb: number
  task: string
  tier: string
  status: "available"
}

interface QuickstartDefaults {
  task: string
  tier: string
  gpu: string
  ramGb: number
  numGpus: number
}

function formatTime(sec: number): string {
  if (sec < 90) return `~${Math.round(sec)} s`
  return `~${Math.round(sec / 60)} min`
}

function InfoTip({ content }: { content: string }) {
  const [show, setShow] = useState(false)
  return (
    <span
      className="relative inline-flex"
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
    >
      <Info className="h-3 w-3 cursor-help" />
      {show && (
        <span className="absolute bottom-full left-0 z-50 mb-1.5 w-56 rounded-lg border border-border bg-white p-2.5 text-left text-xs font-normal leading-relaxed text-muted-foreground shadow-xl">
          {content}
        </span>
      )}
    </span>
  )
}

const ICONS: Record<IconName, typeof Film> = {
  Film,
  Image,
  Sparkles,
  Gamepad2,
}

const DEFAULTS = quickstartData.defaults as QuickstartDefaults
const RAM_OPTIONS = quickstartData.ramOptionsGb as number[]
const GPU_COUNT_OPTIONS = quickstartData.gpuCountOptions as number[]
const HARDWARE = quickstartData.hardware as HardwareOption[]
const HARDWARE_BY_ID = Object.fromEntries(HARDWARE.map((hardware) => [hardware.id, hardware])) as Record<string, HardwareOption>
const PROFILES = quickstartData.profiles as Profile[]
const TASKS = quickstartData.tasks as TaskOption[]
const TIERS = quickstartData.tiers as TierOption[]

function getProfile(task: string, tier: string, gpu: string, vramGb: number): Profile | undefined {
  return PROFILES.find((profile) =>
    profile.task === task &&
    profile.tier === tier &&
    profile.gpu === gpu &&
    profile.vramGb === vramGb
  )
}

export default function FastVideoConfigSelector() {
  const [task, setTask] = useState(DEFAULTS.task)
  const [tier, setTier] = useState(DEFAULTS.tier)
  const [gpu, setGpu] = useState(DEFAULTS.gpu)
  const [vram, setVram] = useState(HARDWARE_BY_ID[DEFAULTS.gpu]?.defaultVramGb ?? HARDWARE[0]?.defaultVramGb ?? 0)
  const [ram, setRam] = useState(DEFAULTS.ramGb)
  const [ngpu, setNgpu] = useState(DEFAULTS.numGpus)
  const [copied, setCopied] = useState(false)

  const hardware = HARDWARE_BY_ID[gpu] ?? HARDWARE[0]
  const config = getProfile(task, tier, gpu, vram)

  useEffect(() => {
    const selectedHardware = HARDWARE_BY_ID[gpu]
    if (!selectedHardware) return
    const options = selectedHardware.vramOptionsGb
    if (!options.includes(vram)) {
      setVram(selectedHardware.defaultVramGb)
    }
  }, [gpu, vram])

  // Report content height to the parent docs page so the embedding iframe can
  // auto-resize — same mechanism as the Advanced Tuning guide.
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
        window.parent.postMessage({ type: "quick-start-guide-height", height }, "*")
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

  const warning = useMemo(() => {
    if (!config) {
      return null
    }
    if (vram < config.vramMin) {
      return `This mode requires ${config.vramMin} GB VRAM. Your selection is ${vram} GB — consider switching to Quick Try.`
    }
    if (ram < config.ramMin) {
      return `This mode requires ${config.ramMin} GB RAM. Your selection is ${ram} GB.`
    }
    return null
  }, [vram, ram, config])

  const command = useMemo(() => {
    if (!config) return ""

    const fps = config.fps ?? 24
    const workloadType = task === "ti2v" ? "i2v" : task

    if (task === "game") {
      const offloadKwargs = [
        config.ditCpuOffload ? "    dit_cpu_offload=True," : null,
        config.ditLayerwiseOffload ? "    dit_layerwise_offload=True," : null,
        config.vaeCpuOffload ? "    vae_cpu_offload=True," : null,
        config.textEncoderCpuOffload ? "    text_encoder_cpu_offload=True," : null,
        config.pinCpuMemory ? "    pin_cpu_memory=True," : null,
        config.useFsdpInference ? "    use_fsdp_inference=True," : null,
      ].filter((line): line is string => Boolean(line))
      const keyboardDim = config.keyboardDim ?? 4
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
        "from fastvideo.models.dits.matrixgame.utils import create_action_presets",
        "",
        `os.environ["FASTVIDEO_ATTENTION_BACKEND"] = "${config.attn}"`,
        "",
        "generator = VideoGenerator.from_pretrained(",
        `    "${config.model}",`,
        `    num_gpus=${ngpu},`,
        ...offloadKwargs,
        ")",
        "",
        `num_frames = ${config.frames}`,
        `actions = create_action_presets(num_frames, keyboard_dim=${keyboardDim})`,
        `grid_sizes = torch.tensor([${Math.floor((config.frames - 1) / 4) + 1}, ${Math.floor(config.h / 8)}, ${Math.floor(config.w / 8)}])`,
        "",
        "generator.generate_video(",
        '    prompt="",',
        '    image_path="./input.png",',
        '    mouse_cond=actions["mouse"].unsqueeze(0),',
        '    keyboard_cond=actions["keyboard"].unsqueeze(0),',
        "    grid_sizes=grid_sizes,",
        `    height=${config.h},`,
        `    width=${config.w},`,
        `    fps=${fps},`,
        "    seed=42,",
        `    num_frames=${config.frames},`,
        `    num_inference_steps=${config.steps},`,
        `    guidance_scale=${config.cfg.toFixed(1)},`,
        '    output_path="outputs/",',
        "    save_video=True,",
        ")",
      ].join("\n")
    }

    const offloadYaml = [
      `      dit: ${Boolean(config.ditCpuOffload)}`,
      `      dit_layerwise: ${Boolean(config.ditLayerwiseOffload)}`,
      `      text_encoder: ${Boolean(config.textEncoderCpuOffload)}`,
      "      image_encoder: false",
      `      vae: ${Boolean(config.vaeCpuOffload)}`,
      `      pin_cpu_memory: ${Boolean(config.pinCpuMemory)}`,
    ]
    const inputsYaml = task === "i2v" || task === "ti2v"
      ? [
          "  inputs:",
          "    image_path: ./input.png",
        ]
      : []
    const fsdpYaml = config.useFsdpInference ? ["    use_fsdp_inference: true"] : []

    return [
      "# Install once:",
      "#   pip install fastvideo",
      "",
      `# ${hardware.label} benchmark: ${formatTime(config.timeSec)} generation time${config.firstRunSec ? `, ${formatTime(config.firstRunSec)} first run including model load` : ""}.`,
      "",
      "cat > fastvideo-generate.yaml <<'YAML'",
      "generator:",
      `  model_path: ${config.model}`,
      "  engine:",
      `    num_gpus: ${ngpu}`,
      "    offload:",
      ...offloadYaml,
      ...fsdpYaml,
      "  pipeline:",
      `    workload_type: ${workloadType}`,
      "",
      "request:",
      '  prompt: "your prompt here"',
      ...inputsYaml,
      "  sampling:",
      `    height: ${config.h}`,
      `    width: ${config.w}`,
      `    fps: ${fps}`,
      "    seed: 42",
      `    num_frames: ${config.frames}`,
      `    num_inference_steps: ${config.steps}`,
      `    guidance_scale: ${config.cfg.toFixed(1)}`,
      "  output:",
      "    output_path: outputs/",
      "    save_video: true",
      "YAML",
      "",
      `FASTVIDEO_ATTENTION_BACKEND=${config.attn} \\`,
      "  fastvideo generate --config fastvideo-generate.yaml",
    ].join("\n")
  }, [config, hardware.label, ngpu, task])

  const copyCommand = async () => {
    if (!command) return
    await navigator.clipboard.writeText(command)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="min-h-screen bg-white text-foreground">
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
                  key={id}
                  onClick={() => setTask(id)}
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

        {/* Step 2: Machine Specs */}
        <section className="mb-8">
          <div className="mb-4 flex items-center gap-2">
            <span className="flex h-6 w-6 items-center justify-center rounded-full bg-primary text-xs font-medium text-primary-foreground">2</span>
            <h2 className="text-sm font-semibold uppercase tracking-wider text-primary">Your machine specs</h2>
          </div>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <div className="rounded-xl border border-border bg-card p-4">
              <div className="mb-2 flex items-center gap-2 text-primary">
                <Monitor className="h-4 w-4" />
                <span className="text-xs font-semibold uppercase tracking-wide">GPU Model</span>
              </div>
              <select
                value={gpu}
                onChange={(e) => {
                  const nextGpu = e.target.value
                  setGpu(nextGpu)
                  setVram(HARDWARE_BY_ID[nextGpu].defaultVramGb)
                }}
                className="w-full rounded-lg border border-border bg-card px-3 py-2 text-sm text-card-foreground outline-none transition-colors focus:border-primary/50"
              >
                {HARDWARE.map((option) => (
                  <option key={option.id} value={option.id}>{option.label}</option>
                ))}
              </select>
            </div>
            <div className="rounded-xl border border-border bg-card p-4">
              <div className="mb-2 flex items-center gap-2 text-primary">
                <Cpu className="h-4 w-4" />
                <span className="text-xs font-semibold uppercase tracking-wide">VRAM (GB)</span>
              </div>
              <select
                value={vram}
                onChange={(e) => setVram(Number(e.target.value))}
                className="w-full rounded-lg border border-border bg-card px-3 py-2 text-sm text-card-foreground outline-none transition-colors focus:border-primary/50"
              >
                {hardware.vramOptionsGb.map((option) => (
                  <option key={option} value={option}>{option} GB</option>
                ))}
              </select>
            </div>
            <div className="rounded-xl border border-border bg-card p-4">
              <div className="mb-2 flex items-center gap-2 text-primary">
                <HardDrive className="h-4 w-4" />
                <span className="text-xs font-semibold uppercase tracking-wide">RAM (GB)</span>
              </div>
              <select
                value={ram}
                onChange={(e) => setRam(Number(e.target.value))}
                className="w-full rounded-lg border border-border bg-card px-3 py-2 text-sm text-card-foreground outline-none transition-colors focus:border-primary/50"
              >
                {RAM_OPTIONS.map((option) => (
                  <option key={option} value={option}>{option} GB</option>
                ))}
              </select>
            </div>
            <div className="rounded-xl border border-border bg-card p-4">
              <div className="mb-2 flex items-center gap-2 text-primary">
                <Layers className="h-4 w-4" />
                <span className="text-xs font-semibold uppercase tracking-wide">Number of GPUs</span>
              </div>
              <select
                value={ngpu}
                onChange={(e) => setNgpu(Number(e.target.value))}
                className="w-full rounded-lg border border-border bg-card px-3 py-2 text-sm text-card-foreground outline-none transition-colors focus:border-primary/50"
              >
                {GPU_COUNT_OPTIONS.map((option) => (
                  <option key={option} value={option}>{option} GPU{option === 1 ? "" : "s"}</option>
                ))}
              </select>
            </div>
          </div>
        </section>

        {/* Step 3: Output Mode */}
        <section className="mb-8">
          <div className="mb-4 flex items-center gap-2">
            <span className="flex h-6 w-6 items-center justify-center rounded-full bg-primary text-xs font-medium text-primary-foreground">3</span>
            <h2 className="text-sm font-semibold uppercase tracking-wider text-primary">Output mode</h2>
          </div>
          <div className="grid gap-3 sm:grid-cols-4">
            {TIERS.map(({ id, name, desc, badge }) => {
              const tierConfig = getProfile(task, id, gpu, vram)
              return (
                <button
                  key={id}
                  onClick={() => setTier(id)}
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
                    {tierConfig ? (
                      <>
                        <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                          <span className="font-semibold">INF</span>
                          <InfoTip content="Inference time only — the compute to generate the video once the model is loaded. The first run takes longer because it also loads the model." />
                          <span>{formatTime(tierConfig.timeSec)}</span>
                        </div>
                        <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                          <span className="font-semibold">VRAM</span>
                          <span>{tierConfig.vram}</span>
                        </div>
                      </>
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
          {warning && (
            <div className="mt-3 rounded-lg border border-destructive/20 bg-destructive/10 px-4 py-3 text-sm text-destructive">
              {warning}
            </div>
          )}
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
            <div className="grid grid-cols-3 divide-x divide-border border-b border-border">
              <div className="p-4">
                <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Model</div>
                <div className="mt-1 font-mono text-sm text-card-foreground">{config?.modelShort ?? "Unavailable"}</div>
              </div>
              <div className="p-4">
                <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Est. Generation Time</div>
                <div className="mt-1 font-mono text-sm text-card-foreground">{config ? formatTime(config.timeSec) : "Unavailable"}</div>
              </div>
              <div className="p-4">
                <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Peak VRAM</div>
                <div className="mt-1 font-mono text-sm text-card-foreground">{config?.vram ?? "Unavailable"}</div>
              </div>
            </div>
            {/* Command Block */}
            <div className="relative p-4">
              {command && (
                <button
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
                  {`${hardware.label} (${vram} GB) is not suitable for ${tier === "quick" ? "Quick Generate" : "High Quality"} on this task.`}
                </div>
              )}
            </div>
          </div>
        </section>
      </div>
    </div>
  )
}
