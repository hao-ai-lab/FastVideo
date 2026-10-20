import { useState } from "react";

export type Validity = "valid" | "invalid" | "loading";
export type Fit = "fits" | "tight" | "exceeds" | "unsupported" | "unknown";
export type Evidence = "recorded" | "none" | "stale";

const validityLabel: Record<Validity, string> = {
  valid: "Configuration valid",
  invalid: "Configuration invalid",
  loading: "Checking configuration",
};
const fitLabel: Record<Fit, string> = {
  fits: "Fits estimate",
  tight: "Tight fit",
  exceeds: "Likely exceeds memory",
  unsupported: "Unsupported combination",
  unknown: "Fit unknown",
};
const evidenceLabel: Record<Evidence, string> = {
  recorded: "Run evidence available",
  none: "No recorded run",
  stale: "Run evidence does not match these settings",
};

/**
 * Three separate facts: the recipe schema accepts the values, a memory estimate
 * covers the configuration, and a recorded run matches it. No positive claim is
 * made until the shared API supplies the corresponding fact.
 */
export default function RecipeFacts({
  validity,
  fit = "unknown",
  evidence = "none",
  subject = "configuration",
  detail,
}: {
  validity: Validity;
  fit?: Fit;
  evidence?: Evidence;
  subject?: "configuration" | "example";
  detail?: string;
}) {
  const [open, setOpen] = useState(false);
  const id = `recipe-facts-${validity}-${fit}-${evidence}`;
  return (
    <div className="recipe-facts">
      <div className="recipe-fact-list" aria-label="Recipe facts">
        <span data-state={validity}>{validity === "valid" ? `Valid ${subject}` : validityLabel[validity]}</span>
        <span data-state={fit}>{fitLabel[fit]}</span>
        <span data-state={evidence}>{evidenceLabel[evidence]}</span>
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
            {validity === "valid"
              ? "Selected values pass the recipe schema and runtime topology checks."
              : validity === "invalid"
                ? "Output is withheld until the highlighted fields are corrected."
                : "Values are still being checked."}
          </p>
          <p>
            <strong>{fitLabel[fit]}.</strong>{" "}
            {fit === "unknown"
              ? "No memory estimate covers this deployment, device, and request yet. This is not a fit claim."
              : "The shared capacity API supplied this estimate; it is not a measured guarantee."}
          </p>
          <p>
            <strong>{evidenceLabel[evidence]}.</strong>{" "}
            {evidence === "none"
              ? "Successful runs on other checkpoints, settings, software revisions, or GPU layouts do not establish this configuration."
              : evidence === "stale"
                ? "A recorded run exists, but edited settings fall outside what it covers."
                : "A recorded run matches the deployment and settings shown."}
          </p>
          {detail && <p>{detail}</p>}
        </div>
      )}
    </div>
  );
}
