import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import ModelHeader from "@/components/cookbook/ModelHeader";
import { sitePath, type DirectoryModel } from "@/lib/cookbook-pages";

/** Existing inference recipes stay runnable while editable serving catalogs are added. */
export default function SourceRecipe({ model }: { model: DirectoryModel }) {
  const [selected, setSelected] = useState(model.recipes[0].id);
  const [mode, setMode] = useState("inference");
  const [yaml, setYaml] = useState("");
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const recipe = model.recipes.find((recipe) => recipe.id === selected)!;
  const yamlSource =
    mode === "serving" ? recipe.serving?.source : /\.ya?ml$/.test(recipe.source) ? recipe.source : undefined;
  const sourceUrl = sitePath(`/assets/cookbook-sources/${recipe.id}-${mode}.yaml`);
  useEffect(() => {
    const controller = new AbortController();
    setYaml("");
    setError("");
    if (yamlSource)
      fetch(sourceUrl, { signal: controller.signal })
        .then(async (response) => {
          if (!response.ok) throw new Error(`YAML could not be loaded: HTTP ${response.status}.`);
          const text = await response.text();
          if (!controller.signal.aborted) setYaml(text);
        })
        .catch((error) => {
          if (!controller.signal.aborted) setError(error.message);
        });
    return () => controller.abort();
  }, [sourceUrl, yamlSource]);
  async function copy(value: string) {
    try {
      await navigator.clipboard.writeText(value);
      setNotice("Copied to clipboard.");
    } catch {
      setNotice("Copy failed. Select and copy the text below.");
    }
  }
  const command =
    mode === "serving" && recipe.serving
      ? `${recipe.serving.env ? `${recipe.serving.env} ` : ""}fastvideo serve --config ${recipe.serving.source}`
      : recipe.command;
  return (
    <div className="cookbook not-prose recipe-detail">
      <ModelHeader
        model={model}
        actions={
          <>
            {yamlSource ? (
              <>
                <Button variant="outline" disabled={!yaml} onClick={() => copy(yaml)}>
                  Copy YAML
                </Button>
                <Button asChild>
                  <a href={sourceUrl} download={yamlSource?.split("/").at(-1)}>
                    Download YAML
                  </a>
                </Button>
              </>
            ) : (
              <Button variant="outline" onClick={() => copy(command)}>
                Copy command
              </Button>
            )}
          </>
        }
      />
      <section className="recipe-command-box" aria-label="Source recipe">
        <div className="recipe-command-toolbar">
          <span className="recipe-unverified">Unverified recipe</span>
          <button type="button" onClick={() => copy(command)}>
            Copy command
          </button>
        </div>
        <div className="recipe-command-content">
          <pre tabIndex={0} aria-label="Inference command">
            <code>{command}</code>
          </pre>
          {yamlSource && (
            <details>
              <summary>View source YAML</summary>
              {error ? (
                <p role="alert">{error}</p>
              ) : (
                <pre tabIndex={0} aria-label="Source YAML">
                  <code>{yaml || "Loading YAML…"}</code>
                </pre>
              )}
            </details>
          )}
        </div>
        <p role="status" className="recipe-copy-status">
          {notice}
        </p>
      </section>
      <details className="recipe-install">
        <summary>Installation and setup</summary>
        {mode === "serving" && recipe.serving && (
          <pre>
            <code>{recipe.serving.install}</code>
          </pre>
        )}
        <a href={sitePath(model.familyPath)}>Model setup guide ↗</a>
      </details>
      <section className="recipe-matrix" aria-label="Recipe options">
        <div className="matrix-row">
          <h2>Recipe</h2>
          <div className="matrix-row-content">
            <div className="option-chips" role="group" aria-label="Recipe variant">
              {model.recipes.map((entry) => (
                <button
                  key={entry.id}
                  type="button"
                  aria-pressed={selected === entry.id}
                  onClick={() => {
                    setSelected(entry.id);
                    setMode("inference");
                    setNotice("");
                    setYaml("");
                  }}
                >
                  {entry.label}
                </button>
              ))}
            </div>
          </div>
        </div>
        <div className="matrix-row">
          <h2>Hardware</h2>
          <div className="matrix-row-content">
            <div className="option-chips">
              {recipe.gpu_types.map((platform) => (
                <span className="option-chip-static" key={platform}>
                  {platform === "Apple Silicon" ? "MLX" : platform}
                </span>
              ))}
            </div>
            <p className="matrix-note">
              {recipe.hardware?.accelerator ??
                (recipe.hardware?.gpu_count
                  ? `${recipe.hardware.gpu_count} GPU${recipe.hardware.gpu_count === 1 ? "" : "s"} in the source recipe`
                  : "See the model guide for resources")}
            </p>
          </div>
        </div>
        <div className="matrix-row">
          <h2>Run mode</h2>
          <div className="matrix-row-content">
            <div className="option-chips" role="group" aria-label="Run mode">
              <button type="button" aria-pressed={mode === "inference"} onClick={() => setMode("inference")}>
                Inference
              </button>
              {recipe.serving && (
                <button type="button" aria-pressed={mode === "serving"} onClick={() => setMode("serving")}>
                  Serving
                </button>
              )}
            </div>
          </div>
        </div>
      </section>
      <div className="recipe-source-links">
        <a href={sitePath(model.familyPath)}>Model guide →</a>
        <a
          href={`https://github.com/hao-ai-lab/FastVideo/blob/main/${mode === "serving" ? recipe.serving?.source : recipe.source}`}
        >
          View source ↗
        </a>
        <span>
          Run from the FastVideo checkout. Source recipe. Interactive YAML configuration is not available for this model
          yet.
        </span>
      </div>
    </div>
  );
}
