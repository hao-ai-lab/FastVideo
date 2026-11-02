import { useId, useState } from "react";
import type { MemoryAssessment } from "@/lib/capacity";

export type Validity = "valid" | "invalid" | "loading";
export type Fit = "fits" | "tight" | "exceeds" | "unsupported" | "unknown";

const validityLabel: Record<Validity, string> = {
  valid: "Configuration valid",
  invalid: "Configuration invalid",
  loading: "Checking configuration",
};
const fitLabel: Record<Fit, string> = {
  fits: "Fits estimate",
  tight: "Tight fit",
  exceeds: "Exceeds memory",
  unsupported: "Unsupported",
  unknown: "Memory not estimated",
};
const weightsLabel: Record<Fit, string> = {
  fits: "Weights fit",
  tight: "Weights fit, little headroom",
  exceeds: "Weights exceed memory",
  unsupported: "Weights unsupported",
  unknown: "Weights not estimated",
};

/**
 * Two facts, each with its own evidence: the recipe schema accepts the values,
 * and the analytical estimate covers the selected device. Neither is a promise
 * that the recipe runs.
 */
export default function RecipeFacts({
  validity,
  memory,
  subject = "configuration",
  validityText,
  validityDetail,
  detail,
}: {
  validity: Validity;
  memory?: MemoryAssessment;
  subject?: "configuration" | "example";
  validityText?: string;
  validityDetail?: string;
  detail?: string;
}) {
  const [open, setOpen] = useState(false);
  const id = useId();
  const fit: Fit = memory?.status ?? "unknown";
  const estimated = fit === "fits" || fit === "tight" || fit === "exceeds";
  const weightsBasis = memory?.basis === "weights";
  const label = weightsBasis ? weightsLabel[fit] : fitLabel[fit];
  return (
    <div className="recipe-facts">
      <div className="recipe-fact-list" aria-label="Recipe checks">
        <span data-state={validity}>
          {validity === "valid" ? (validityText ?? `Valid ${subject}`) : validityLabel[validity]}
        </span>
        <span data-state={fit}>{label}</span>
      </div>
      <button
        type="button"
        className="recipe-facts-toggle"
        aria-expanded={open}
        aria-controls={id}
        onClick={() => setOpen(!open)}
      >
        Details
      </button>
      {open && (
        <div className="recipe-facts-detail" id={id}>
          <p>
            <strong>{validityLabel[validity]}.</strong>{" "}
            {validityDetail ??
              (validity === "valid"
                ? "These values pass the recipe schema and runtime topology checks."
                : validity === "invalid"
                  ? "Output is hidden until the highlighted fields are fixed."
                  : "Values are still being checked.")}
          </p>
          <p>
            <strong>{label}.</strong>{" "}
            {weightsBasis
              ? `${memory?.reason ?? ""} Resident weights need ${memory?.requiredGiB?.toFixed(1)} GiB of ${memory?.capacityGiB?.toFixed(1)} GiB available.`
              : estimated
                ? `The analytical estimate needs ${memory?.requiredGiB?.toFixed(1)} GiB of ${memory?.capacityGiB?.toFixed(1)} GiB available. It models weights, encoders, and the configured workload. It is not a measured guarantee.`
                : (memory?.reason ??
                  "FastVideo has no capacity model for this checkpoint, so it does not estimate memory use here.")}
          </p>
          {detail && <p>{detail}</p>}
        </div>
      )}
    </div>
  );
}
