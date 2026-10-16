import { useState } from "react";

/** A UI experiment, deliberately separated from the serving schema and generated YAML. */
export default function ComponentPlan({ onDraftChange }: { onDraftChange?: (draft: boolean) => void }) {
  const [selection, setSelection] = useState<Record<string, string>>({});
  const groups = [
    { name: "Weights", values: ["Recipe default", "FP8", "NVFP4"] },
    { name: "Text encoder", values: ["Recipe default", "NVFP4 encoder"] },
    { name: "Video decoder", values: ["Recipe default", "FP16 VAE", "NVFP4 VAE"] },
  ];
  return (
    <details className="component-plan">
      <summary>
        Explore components <span>Design preview</span>
      </summary>
      <div className="component-plan-fields">
        {groups.map((group) => (
          <div key={group.name}>
            <span className="component-plan-label">{group.name}</span>
            <div className="option-chips" role="group" aria-label={group.name}>
              {group.values.map((value) => (
                <button
                  key={value}
                  type="button"
                  aria-pressed={(selection[group.name] ?? "Recipe default") === value}
                  onClick={() => {
                    const next = { ...selection, [group.name]: value };
                    setSelection(next);
                    onDraftChange?.(Object.values(next).some((value) => value !== "Recipe default"));
                  }}
                >
                  {value}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>
      <p>
        Preview only. Alternate components need a compatible recipe; some releases are planned. These choices do not
        change YAML.
      </p>
      {Object.values(selection).some((value) => value !== "Recipe default") && (
        <button
          type="button"
          className="component-plan-reset"
          onClick={() => {
            setSelection({});
            onDraftChange?.(false);
          }}
        >
          Restore recipe components
        </button>
      )}
    </details>
  );
}
