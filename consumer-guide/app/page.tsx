"use client"

import { useState, useMemo } from "react"
import { Monitor, Cpu, HardDrive, Layers, Clock, Copy, Check, Film, Image, Sparkles, Gamepad2, Settings2 } from "lucide-react"
import Link from "next/link"

type Task = "t2v" | "i2v" | "ti2v" | "action"
type Tier = "quick" | "daily" | "best"

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
  time: string
  vram: string
  offload: boolean
  extra?: string
}

const CONFIGS: Record<Task, Record<Tier, Config>> = {
  t2v: {
    quick: { model: "FastVideo/FastWan2.1-T2V-1.3B-Diffusers", modelShort: "FastWan2.1-T2V-1.3B", attn: "VIDEO_SPARSE_ATTN", h: 480, w: 832, frames: 49, steps: 20, cfg: 5.0, vramMin: 8, ramMin: 16, time: "~60 s", vram: "~16 GB", offload: false },
    daily: { model: "FastVideo/FastWan2.1-T2V-1.3B-Diffusers", modelShort: "FastWan2.1-T2V-1.3B", attn: "FLASH_ATTN", h: 720, w: 1280, frames: 81, steps: 30, cfg: 5.0, vramMin: 20, ramMin: 32, time: "~3 min", vram: "~22 GB", offload: false },
    best: { model: "Wan-AI/Wan2.1-T2V-14B-Diffusers", modelShort: "Wan2.1-T2V-14B", attn: "FLASH_ATTN", h: 720, w: 1280, frames: 81, steps: 50, cfg: 6.0, vramMin: 16, ramMin: 40, time: "~15 min", vram: "~18 GB + ~40 GB RAM", offload: true }
  },
  i2v: {
    quick: { model: "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", modelShort: "Wan2.1-I2V-14B-480P", attn: "FLASH_ATTN", h: 480, w: 832, frames: 49, steps: 20, cfg: 5.0, vramMin: 16, ramMin: 32, time: "~2 min", vram: "~18 GB", offload: true, extra: "--image ./input.png" },
    daily: { model: "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", modelShort: "Wan2.1-I2V-14B-480P", attn: "FLASH_ATTN", h: 480, w: 832, frames: 81, steps: 30, cfg: 5.0, vramMin: 16, ramMin: 32, time: "~5 min", vram: "~20 GB", offload: true, extra: "--image ./input.png" },
    best: { model: "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers", modelShort: "Wan2.1-I2V-14B-720P", attn: "FLASH_ATTN", h: 720, w: 1280, frames: 81, steps: 50, cfg: 6.0, vramMin: 16, ramMin: 40, time: "~20 min", vram: "~20 GB + ~40 GB RAM", offload: true, extra: "--image ./input.png" }
  },
  ti2v: {
    quick: { model: "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers", modelShort: "FastWan2.2-TI2V-5B", attn: "FLASH_ATTN", h: 480, w: 832, frames: 49, steps: 20, cfg: 5.0, vramMin: 16, ramMin: 32, time: "~2 min", vram: "~18 GB", offload: false, extra: "--image ./input.png" },
    daily: { model: "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers", modelShort: "FastWan2.2-TI2V-5B", attn: "FLASH_ATTN", h: 720, w: 1280, frames: 81, steps: 30, cfg: 5.0, vramMin: 20, ramMin: 32, time: "~5 min", vram: "~22 GB", offload: false, extra: "--image ./input.png" },
    best: { model: "Wan-AI/Wan2.2-TI2V-5B-Diffusers", modelShort: "Wan2.2-TI2V-5B", attn: "SAGE_ATTN", h: 720, w: 1280, frames: 81, steps: 50, cfg: 6.0, vramMin: 20, ramMin: 40, time: "~15 min", vram: "~22 GB + ~40 GB RAM", offload: true, extra: "--image ./input.png" }
  },
  action: {
    quick: { model: "FastVideo/Matrix-Game-2.0-Base-Diffusers", modelShort: "Matrix-Game-2.0-Base", attn: "FLASH_ATTN", h: 480, w: 832, frames: 49, steps: 20, cfg: 5.0, vramMin: 16, ramMin: 32, time: "~2 min", vram: "~18 GB", offload: false, extra: "--image ./input.png --action forward" },
    daily: { model: "FastVideo/Matrix-Game-2.0-Base-Diffusers", modelShort: "Matrix-Game-2.0-Base", attn: "FLASH_ATTN", h: 720, w: 1280, frames: 81, steps: 30, cfg: 5.0, vramMin: 20, ramMin: 32, time: "~5 min", vram: "~22 GB", offload: false, extra: "--image ./input.png --action forward" },
    best: { model: "FastVideo/Matrix-Game-2.0-GTA-Diffusers", modelShort: "Matrix-Game-2.0-GTA", attn: "FLASH_ATTN", h: 720, w: 1280, frames: 81, steps: 50, cfg: 6.0, vramMin: 20, ramMin: 40, time: "~15 min", vram: "~20 GB + ~40 GB RAM", offload: true, extra: "--image ./input.png --action forward" }
  }
}

const TASKS: { id: Task; label: string; sub: string; icon: typeof Film }[] = [
  { id: "t2v", label: "Text to Video", sub: "T2V", icon: Film },
  { id: "i2v", label: "Image to Video", sub: "I2V", icon: Image },
  { id: "ti2v", label: "Text + Image", sub: "TI2V", icon: Sparkles },
  { id: "action", label: "Action Control", sub: "Matrix Game", icon: Gamepad2 },
]

const TIERS: { id: Tier; name: string; desc: string; badge?: string }[] = [
  { id: "quick", name: "Quick Try", desc: "Fast preview. Lower resolution, fewer steps." },
  { id: "daily", name: "Daily Use", desc: "Balanced quality and speed.", badge: "Recommended" },
  { id: "best", name: "Best Quality", desc: "14B model, max fidelity. Requires offloading." },
]

export default function FastVideoConfigSelector() {
  const [task, setTask] = useState<Task>("t2v")
  const [tier, setTier] = useState<Tier>("daily")
  const [gpu, setGpu] = useState("4090")
  const [vram, setVram] = useState(24)
  const [ram, setRam] = useState(64)
  const [ngpu, setNgpu] = useState(1)
  const [copied, setCopied] = useState(false)

  const config = CONFIGS[task][tier]

  const warning = useMemo(() => {
    if (vram < config.vramMin) {
      return `This mode requires ${config.vramMin} GB VRAM. Your selection is ${vram} GB — consider switching to Quick Try.`
    }
    if (ram < config.ramMin) {
      return `This mode requires ${config.ramMin} GB RAM. Your selection is ${ram} GB.`
    }
    return null
  }, [vram, ram, config])

  const command = useMemo(() => {
    let cmd = `# 1. install\npip install fastvideo\n\n`
    cmd += `# 2. download model\nhuggingface-cli download ${config.model} \\\n  --local-dir ./models/${config.modelShort}\n\n`
    cmd += `# 3. run\nfastvideo generate \\\n`
    cmd += `  --model ./models/${config.modelShort} \\\n`
    cmd += `  --attention ${config.attn} \\\n`
    cmd += `  --height ${config.h} --width ${config.w} \\\n`
    cmd += `  --num_frames ${config.frames} \\\n`
    cmd += `  --num_inference_steps ${config.steps} \\\n`
    cmd += `  --guidance_scale ${config.cfg.toFixed(1)}`
    if (config.offload) {
      cmd += ` \\\n  --dit_cpu_offload \\\n  --vae_cpu_offload \\\n  --pin_cpu_memory`
    }
    if (config.extra) {
      cmd += ` \\\n  ${config.extra}`
    }
    cmd += ` \\\n  --prompt "your prompt here"`
    return cmd
  }, [config])

  const copyCommand = async () => {
    await navigator.clipboard.writeText(command)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      {/* Header */}
      <div className="border-b border-white/5 bg-slate-900/50 backdrop-blur-sm">
        <div className="mx-auto max-w-4xl px-6 py-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <img 
                src="https://hebbkx1anhila5yf.public.blob.vercel-storage.com/image-xVE4qNY0zoz6VgwQkgqPAbcJG4rfbP.png" 
                alt="FastVideo Logo" 
                className="h-10 w-auto"
              />
              <div>
                <h1 className="text-lg font-semibold text-white">Config Selector</h1>
                <p className="text-sm text-slate-300">Choose your hardware and task to get a recommended configuration</p>
              </div>
            </div>
            <Link 
              href="/tuning"
              className="flex items-center gap-2 text-sm text-slate-400 hover:text-sky-400 transition-colors"
            >
              <Settings2 className="h-4 w-4" />
              Advanced Tuning
            </Link>
          </div>
        </div>
      </div>

      <div className="mx-auto max-w-4xl px-6 py-8">
        {/* Step 1: Task Selection */}
        <section className="mb-8">
          <div className="mb-4 flex items-center gap-2">
            <span className="flex h-6 w-6 items-center justify-center rounded-full bg-sky-500/20 text-xs font-medium text-sky-400">1</span>
            <h2 className="text-sm font-semibold uppercase tracking-wider text-sky-400">What do you want to make?</h2>
          </div>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            {TASKS.map(({ id, label, sub, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setTask(id)}
                className={`group relative rounded-xl border p-4 text-left transition-all duration-200 ${
                  task === id
                    ? "border-sky-500/50 bg-sky-500/10 shadow-lg shadow-sky-500/10"
                    : "border-white/5 bg-white/[0.02] hover:border-white/10 hover:bg-white/[0.04]"
                }`}
              >
                <div className={`mb-3 flex h-10 w-10 items-center justify-center rounded-lg transition-colors ${
                  task === id ? "bg-sky-500/20 text-sky-400" : "bg-white/5 text-slate-500 group-hover:text-slate-400"
                }`}>
                  <Icon className="h-5 w-5" />
                </div>
                <div className={`font-medium transition-colors ${task === id ? "text-white" : "text-slate-300"}`}>
                  {label}
                </div>
                <div className="mt-0.5 text-xs text-slate-500">{sub}</div>
              </button>
            ))}
          </div>
        </section>

        {/* Step 2: Machine Specs */}
        <section className="mb-8">
          <div className="mb-4 flex items-center gap-2">
            <span className="flex h-6 w-6 items-center justify-center rounded-full bg-sky-500/20 text-xs font-medium text-sky-400">2</span>
            <h2 className="text-sm font-semibold uppercase tracking-wider text-sky-400">Your machine specs</h2>
          </div>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <div className="rounded-xl border border-white/5 bg-white/[0.02] p-4">
              <div className="mb-2 flex items-center gap-2 text-sky-400">
                <Monitor className="h-4 w-4" />
                <span className="text-xs font-semibold uppercase tracking-wide">GPU Model</span>
              </div>
              <select
                value={gpu}
                onChange={(e) => setGpu(e.target.value)}
                className="w-full rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-sm text-white outline-none transition-colors focus:border-sky-500/50"
              >
                <option value="4090">RTX 4090</option>
                <option value="4080">RTX 4080</option>
                <option value="3090">RTX 3090</option>
                <option value="3080">RTX 3080</option>
                <option value="other">Other</option>
              </select>
            </div>
            <div className="rounded-xl border border-white/5 bg-white/[0.02] p-4">
              <div className="mb-2 flex items-center gap-2 text-sky-400">
                <Cpu className="h-4 w-4" />
                <span className="text-xs font-semibold uppercase tracking-wide">VRAM (GB)</span>
              </div>
              <select
                value={vram}
                onChange={(e) => setVram(Number(e.target.value))}
                className="w-full rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-sm text-white outline-none transition-colors focus:border-sky-500/50"
              >
                <option value={24}>24 GB</option>
                <option value={16}>16 GB</option>
                <option value={12}>12 GB</option>
                <option value={10}>10 GB</option>
                <option value={8}>8 GB</option>
              </select>
            </div>
            <div className="rounded-xl border border-white/5 bg-white/[0.02] p-4">
              <div className="mb-2 flex items-center gap-2 text-sky-400">
                <HardDrive className="h-4 w-4" />
                <span className="text-xs font-semibold uppercase tracking-wide">RAM (GB)</span>
              </div>
              <select
                value={ram}
                onChange={(e) => setRam(Number(e.target.value))}
                className="w-full rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-sm text-white outline-none transition-colors focus:border-sky-500/50"
              >
                <option value={128}>128 GB</option>
                <option value={64}>64 GB</option>
                <option value={32}>32 GB</option>
                <option value={16}>16 GB</option>
              </select>
            </div>
            <div className="rounded-xl border border-white/5 bg-white/[0.02] p-4">
              <div className="mb-2 flex items-center gap-2 text-sky-400">
                <Layers className="h-4 w-4" />
                <span className="text-xs font-semibold uppercase tracking-wide">Number of GPUs</span>
              </div>
              <select
                value={ngpu}
                onChange={(e) => setNgpu(Number(e.target.value))}
                className="w-full rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-sm text-white outline-none transition-colors focus:border-sky-500/50"
              >
                <option value={1}>1 GPU</option>
                <option value={2}>2 GPUs</option>
                <option value={4}>4 GPUs</option>
              </select>
            </div>
          </div>
        </section>

        {/* Step 3: Output Mode */}
        <section className="mb-8">
          <div className="mb-4 flex items-center gap-2">
            <span className="flex h-6 w-6 items-center justify-center rounded-full bg-sky-500/20 text-xs font-medium text-sky-400">3</span>
            <h2 className="text-sm font-semibold uppercase tracking-wider text-sky-400">Output mode</h2>
          </div>
          <div className="grid gap-3 sm:grid-cols-3">
            {TIERS.map(({ id, name, desc, badge }) => {
              const tierConfig = CONFIGS[task][id]
              return (
                <button
                  key={id}
                  onClick={() => setTier(id)}
                  className={`group relative rounded-xl border p-4 text-left transition-all duration-200 ${
                    tier === id
                      ? "border-sky-500/50 bg-sky-500/10 shadow-lg shadow-sky-500/10"
                      : "border-white/5 bg-white/[0.02] hover:border-white/10 hover:bg-white/[0.04]"
                  }`}
                >
                  {badge && (
                    <span className="mb-2 inline-block rounded-full bg-sky-500/20 px-2.5 py-0.5 text-[10px] font-medium text-sky-400">
                      {badge}
                    </span>
                  )}
                  <div className={`font-medium transition-colors ${tier === id ? "text-white" : "text-slate-300"}`}>
                    {name}
                  </div>
                  <p className="mt-1 text-xs leading-relaxed text-slate-500">{desc}</p>
                  <div className="mt-3 space-y-1 border-t border-white/5 pt-3">
                    <div className="flex items-center gap-1.5 text-xs text-slate-500">
                      <Clock className="h-3 w-3" />
                      <span>{tierConfig.time}</span>
                    </div>
                    <div className="flex items-center gap-1.5 text-xs text-slate-500">
                      <Cpu className="h-3 w-3" />
                      <span>{tierConfig.vram}</span>
                    </div>
                  </div>
                </button>
              )
            })}
          </div>
          {warning && (
            <div className="mt-3 rounded-lg border border-amber-500/20 bg-amber-500/10 px-4 py-3 text-sm text-amber-400">
              {warning}
            </div>
          )}
        </section>

        {/* Output Configuration */}
        <section>
          <div className="mb-4 flex items-center gap-2">
            <span className="flex h-6 w-6 items-center justify-center rounded-full bg-emerald-500/20 text-xs font-medium text-emerald-400">✓</span>
            <h2 className="text-sm font-semibold uppercase tracking-wider text-emerald-400">Your configuration</h2>
          </div>
          <div className="overflow-hidden rounded-2xl border border-white/5 bg-white/[0.02]">
            {/* Meta Cards */}
            <div className="grid grid-cols-3 divide-x divide-white/5 border-b border-white/5">
              <div className="p-4">
                <div className="text-xs font-semibold uppercase tracking-wide text-slate-300">Model</div>
                <div className="mt-1 font-mono text-sm text-white">{config.modelShort}</div>
              </div>
              <div className="p-4">
                <div className="text-xs font-semibold uppercase tracking-wide text-slate-300">Est. Generation Time</div>
                <div className="mt-1 font-mono text-sm text-white">{config.time}</div>
              </div>
              <div className="p-4">
                <div className="text-xs font-semibold uppercase tracking-wide text-slate-300">Peak VRAM</div>
                <div className="mt-1 font-mono text-sm text-white">{config.vram}</div>
              </div>
            </div>
            {/* Command Block */}
            <div className="relative p-4">
              <button
                onClick={copyCommand}
                className="absolute right-4 top-4 flex items-center gap-2 rounded-lg border border-white/10 bg-white/5 px-4 py-2 text-sm font-medium text-slate-300 transition-all hover:border-sky-500/30 hover:bg-sky-500/10 hover:text-sky-400"
              >
                {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                {copied ? "Copied" : "Copy"}
              </button>
              <pre className="overflow-x-auto font-mono text-sm leading-relaxed text-slate-300">
                {command}
              </pre>
            </div>
          </div>
        </section>
      </div>
    </div>
  )
}
