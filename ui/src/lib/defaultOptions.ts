// SPDX-License-Identifier: Apache-2.0

export interface DefaultOptions {
  numInferenceSteps: number;
  numFrames: number;
  height: number;
  width: number;
  guidanceScale: number;
  seed: number;
  numGpus: number;
  ditCpuOffload: boolean;
  textEncoderCpuOffload: boolean;
  useFsdpInference: boolean;
}

export const DEFAULT_OPTIONS: DefaultOptions = {
  numInferenceSteps: 50,
  numFrames: 81,
  height: 480,
  width: 832,
  guidanceScale: 5.0,
  seed: 1024,
  numGpus: 1,
  ditCpuOffload: false,
  textEncoderCpuOffload: false,
  useFsdpInference: false,
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
