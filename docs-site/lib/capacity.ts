import { useEffect, useState } from "react";
// @ts-expect-error The shared ES module has no TypeScript declarations.
import { loadCapacityIndex, loadCapacityModel } from "../../docs/js/cookbook-capacity.mjs";
export { assessRecipe, assessWeightsRecipe, weightsFor } from "./memory-assessment";
export type { MemoryAssessment } from "./memory-assessment";

interface CapacityModel {
  id: string;
  name?: string;
  examples?: { deployment_id?: string }[];
  reference_runs?: unknown[];
  [key: string]: unknown;
}

interface CapacityState {
  status: "loading" | "ready" | "unavailable" | "error";
  model?: CapacityModel;
  error?: string;
}

const CAPACITY_INDEX = `${import.meta.env.BASE_URL.replace(/\/$/, "")}/assets/cookbook-config/capacity/index.json`;

/**
 * Load the capacity model for one cookbook entry. Most recipes do not have one;
 * the hook stays `unavailable` and the UI says so instead of guessing.
 */
export function useCapacityModel(modelId: string | undefined): CapacityState & { retry: () => void } {
  const [state, setState] = useState<CapacityState>({ status: "loading" });
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    if (!modelId) {
      setState({ status: "unavailable" });
      return;
    }
    const controller = new AbortController();
    setState({ status: "loading" });
    loadCapacityIndex(new URL(CAPACITY_INDEX, window.location.origin).href, { signal: controller.signal })
      .then(async (index: { models: { id: string }[] }) => {
        if (!index.models.some((model) => model.id === modelId)) {
          if (!controller.signal.aborted) setState({ status: "unavailable" });
          return;
        }
        const model = (await loadCapacityModel(index, modelId, { signal: controller.signal })) as CapacityModel;
        if (!controller.signal.aborted) setState({ status: "ready", model });
      })
      .catch((error: unknown) => {
        if (!controller.signal.aborted)
          setState({ status: "error", error: error instanceof Error ? error.message : "Unable to load memory data." });
      });
    return () => controller.abort();
  }, [modelId, attempt]);
  return { ...state, retry: () => setAttempt((value) => value + 1) };
}
