// SPDX-License-Identifier: Apache-2.0

export interface DefaultOptions {
  defaultModelId: string;
  numInferenceSteps: number;
  numFrames: number;
  height: number;
  width: number;
  guidanceScale: number;
  guidanceRescale: number;
  fps: number;
  seed: number;
  numGpus: number;
  ditCpuOffload: boolean;
  textEncoderCpuOffload: boolean;
  vaeCpuOffload: boolean;
  imageEncoderCpuOffload: boolean;
  useFsdpInference: boolean;
  enableTorchCompile: boolean;
  vsaSparsity: number;
  tpSize: number;
  spSize: number;
}

export const DEFAULT_OPTIONS: DefaultOptions = {
  defaultModelId: "",
  numInferenceSteps: 50,
  numFrames: 81,
  height: 480,
  width: 832,
  guidanceScale: 5.0,
  guidanceRescale: 0,
  fps: 24,
  seed: 1024,
  numGpus: 1,
  ditCpuOffload: false,
  textEncoderCpuOffload: false,
  vaeCpuOffload: false,
  imageEncoderCpuOffload: false,
  useFsdpInference: false,
  enableTorchCompile: false,
  vsaSparsity: 0,
  tpSize: -1,
  spSize: -1,
};

const STORAGE_KEY = "fastvideo-default-options";

export function loadDefaultOptions(): DefaultOptions {
  if (typeof window === "undefined") {
    return DEFAULT_OPTIONS;
  }
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) return DEFAULT_OPTIONS;
    const parsed = JSON.parse(stored) as Partial<DefaultOptions>;
    return { ...DEFAULT_OPTIONS, ...parsed };
  } catch {
    return DEFAULT_OPTIONS;
  }
}

export function saveDefaultOptions(options: DefaultOptions): void {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(options));
  } catch {
    // Ignore storage errors
  }
}
