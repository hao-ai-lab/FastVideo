import { useId, useState } from "react";
import type { MemoryAssessment } from "@/lib/capacity";

export type Validity = "valid" | "invalid" | "loading";
export type Fit = "fits" | "tight" | "exceeds" | "unsupported" | "unknown";

const validityLabel: Record<Validity, string> = {
  valid: "Schema checks passed",
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
/**
 * Two facts, each with its own evidence: the recipe schema accepts the values,
 * and the analytical estimate covers the selected device. Neither is a promise
 * that the recipe runs.
 */
export default function RecipeFacts({
  validity,
  memory,
  validityText,
  validityDetail,
  detail,
  retryMemory,
}: {
  validity: Validity;
  memory?: MemoryAssessment;
  subject?: "configuration" | "example";
  validityText?: string;
  validityDetail?: string;
  detail?: string;
  retryMemory?: () => void;
}) {
  const [open, setOpen] = useState(false);
  const id = useId();
  const fit: Fit = memory?.status ?? "unknown";
  const estimated = fit === "fits" || fit === "tight" || fit === "exceeds";
  const label = fitLabel[fit];
  return (
    <div className="recipe-facts">
      <div className="recipe-fact-list" aria-label="Recipe checks">
        <span data-state={validity}>
          {validity === "valid" ? (validityText ?? validityLabel.valid) : validityLabel[validity]}
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
      {memory?.reason && (
        <p className="matrix-note" role="status">
          {memory.reason}
        </p>
      )}
      {memory?.basis === "checkpoint" && (
        <div className="recipe-memory-breakdown">
          <p>
            Known checkpoint data: {memory.checkpointGiB?.toFixed(1)} GiB. Device budget:{" "}
            {memory.capacityGiB?.toFixed(1)} GiB.
          </p>
          <dl>
            {Object.entries(memory.parts ?? {}).map(([name, size]) => (
              <div key={name}>
                <dt>{name.replaceAll("_", " ")}</dt>
                <dd>{size.toFixed(1)} GiB</dd>
              </div>
            ))}
          </dl>
        </div>
      )}
      {retryMemory && (
        <button type="button" onClick={retryMemory}>
          Retry memory data
        </button>
      )}
      {open && (
        <div className="recipe-facts-detail" id={id}>
          <p>
            <strong>{validityLabel[validity]}.</strong>{" "}
            {validityDetail ??
              (validity === "valid"
                ? "These values pass the recipe schema and runtime topology checks. Model loading, selected components and GPU execution have not been verified."
                : validity === "invalid"
                  ? "Output is hidden until the highlighted fields are fixed."
                  : "Values are still being checked.")}
          </p>
          {memory?.notes?.map((note) => (
            <p key={note}>{note}</p>
          ))}
          {estimated && (
            <p>
              <strong>{label}.</strong>{" "}
              {`The ${memory?.basis === "measured" ? "recorded peak" : "analytical estimate"} needs ${memory?.requiredGiB?.toFixed(1)} GiB of ${memory?.capacityGiB?.toFixed(1)} GiB available. This applies only to the covered execution settings.`}
            </p>
          )}
          {detail && <p>{detail}</p>}
        </div>
      )}
    </div>
  );
}
