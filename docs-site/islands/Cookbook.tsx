import { useEffect, useId, useState, type ReactNode } from "react";
import { Tabs } from "radix-ui";
import { modelDirectory, recipePages, sitePath, type DirectoryModel } from "@/lib/cookbook-pages";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import SourceRecipe from "./SourceRecipe";
import HardwarePicker from "@/components/cookbook/HardwarePicker";
import ModelHeader from "@/components/cookbook/ModelHeader";
import RecipeFacts from "@/components/cookbook/RecipeFacts";
import { configAndLaunch, inferenceOutput, sdkCommand } from "@/lib/cookbook-commands";
import {
  api,
  guideUrl,
  parseField,
  types,
  type Catalog,
  type Field,
  type Index,
  type Result,
  type Value,
} from "@/lib/cookbook";

export const client = "load";

function groupTitle(path: string) {
  const titles: Record<string, string> = {
    server: "Server",
    "generator.engine": "GPU resources",
    generator: "Checkpoint settings",
    "generator.engine.parallelism": "Parallelism",
    "generator.engine.offload": "CPU offload",
    "generator.engine.compile": "Compilation",
    "default_request.sampling": "Request defaults",
    streaming: "Streaming",
  };
  return titles[path] ?? path.split(".").at(-1)?.replaceAll("_", " ");
}

function errorText(error: unknown) {
  return error instanceof Error ? error.message : "Unable to load this recipe.";
}

function FieldControl({
  field,
  edits,
  update,
  failure,
}: {
  field: Field;
  edits: Record<string, Value>;
  update: (path: string, raw: string | undefined, field: Field) => void;
  failure?: string;
}) {
  const id = useId();
  const kind = types(field.schema);
  const changed = Object.hasOwn(edits, field.path);
  const value = changed ? edits[field.path] : field.value;
  const label = field.schema.title ?? field.path;
  const choices = field.schema.enum ?? (kind.includes("boolean") ? [true, false] : undefined);
  const nullable = kind.includes("null");
  // Keep unfinished numeric and JSON input visible while output is invalid.
  const [raw, setRaw] = useState<string | undefined>();
  const displayed = raw ?? (value == null ? "" : typeof value === "string" ? value : JSON.stringify(value));
  const props = {
    id,
    "aria-describedby": `${id}-hint${failure ? ` ${id}-error` : ""}`,
    "aria-invalid": !!failure,
  };
  function edit(text: string) {
    setRaw(text);
    update(field.path, text, field);
  }
  return (
    <div className="matrix-field">
      <div className="matrix-field-label">
        {choices ? <span>{label}</span> : <label htmlFor={id}>{label}</label>}
        {changed && (
          <button
            type="button"
            aria-label={`Use baseline for ${label}`}
            onClick={() => {
              setRaw(undefined);
              update(field.path, undefined, field);
            }}
          >
            Reset
          </button>
        )}
      </div>
      {choices ? (
        <div role="group" aria-label={label} className="option-chips">
          {value === undefined && (
            <button type="button" aria-pressed={!changed} onClick={() => update(field.path, undefined, field)}>
              Default
            </button>
          )}
          {choices.map((choice) => (
            <button
              key={JSON.stringify(choice)}
              type="button"
              aria-pressed={value === choice}
              onClick={() => update(field.path, JSON.stringify(choice), field)}
            >
              {typeof choice === "boolean" ? (choice ? "On" : "Off") : String(choice)}
            </button>
          ))}
          {nullable && !choices.includes(null) && (
            <button
              type="button"
              aria-pressed={value === null}
              onClick={() => update(field.path, "null", { ...field, schema: { ...field.schema, enum: [null] } })}
            >
              null
            </button>
          )}
        </div>
      ) : kind.includes("object") || kind.includes("array") ? (
        <Textarea
          {...props}
          value={displayed}
          onChange={(event) => edit(event.target.value)}
          placeholder="Inherited; enter JSON"
        />
      ) : (
        <div className={field.path === "generator.engine.num_gpus" ? "resource-stepper" : ""}>
          {field.path === "generator.engine.num_gpus" && (
            <button
              type="button"
              aria-label="Remove a GPU"
              disabled={Number(value) <= (field.schema.minimum ?? 1)}
              onClick={() => edit(String(Number(value) - 1))}
            >
              −
            </button>
          )}
          <Input
            {...props}
            type={kind.includes("integer") || kind.includes("number") ? "number" : "text"}
            value={displayed}
            placeholder={value === undefined ? "Inherited" : value === null ? "null" : ""}
            min={field.schema.minimum}
            max={field.schema.maximum}
            step={field.schema.multipleOf ?? (kind.includes("integer") ? 1 : "any")}
            onChange={(event) => edit(event.target.value)}
          />
          {field.path === "generator.engine.num_gpus" && (
            <button
              type="button"
              aria-label="Add a GPU"
              disabled={Number(value) >= (field.schema.maximum ?? Infinity)}
              onClick={() => edit(String(Number(value) + 1))}
            >
              +
            </button>
          )}
        </div>
      )}
      {nullable && !choices && (
        <button
          className="matrix-null"
          type="button"
          aria-label={`Set ${label} to null`}
          onClick={() => {
            setRaw(undefined);
            update(field.path, "null", { ...field, schema: { ...field.schema, enum: [null] } });
          }}
        >
          Use null
        </button>
      )}
      <p id={`${id}-hint`} className="sr-only">
        {field.schema.description}
        {value === undefined ? " Inherited from the runtime." : ""}
      </p>
      {failure && (
        <p id={`${id}-error`} className="matrix-field-error" role="alert">
          {failure}
        </p>
      )}
    </div>
  );
}

function Editor({
  catalog,
  deployments,
  selectRuntime,
}: {
  catalog: Catalog;
  deployments: Index["models"][number]["deployments"];
  selectRuntime: (id: string) => void;
}) {
  const [advanced, setAdvanced] = useState(false);
  const [runMode, setRunMode] = useState("serving");
  const [sourceImage, setSourceImage] = useState("");
  const sourceImageId = useId();
  const [showEnv, setShowEnv] = useState(false);
  const [showClient, setShowClient] = useState(false);
  const [outputTab, setOutputTab] = useState("commands");
  const [hardware, setHardware] = useState(() => {
    const requested = new URL(window.location.href).searchParams.get("hardware");
    return requested && requested !== "unspecified" ? requested : "";
  });
  const [edits, setEdits] = useState<Record<string, Value>>({});
  const [parseErrors, setParseErrors] = useState<Record<string, string>>({});
  const [reset, setReset] = useState(0);
  const [notice, setNotice] = useState("");
  let result: Result | undefined;
  let output: ReturnType<typeof inferenceOutput> | undefined;
  let failure = "";
  let failedPath: string | undefined;
  try {
    if (Object.keys(parseErrors).length) throw new Error("Correct the highlighted fields to generate a configuration.");
    result = api.resolveConfig(catalog, edits);
    output =
      runMode === "serving" ? result : inferenceOutput(result, catalog.workload === "i2v" ? sourceImage : undefined);
  } catch (error) {
    failure = errorText(error);
    failedPath = (error as { path?: string }).path;
  }
  const fields = api.getOptions(catalog, catalog.base_config, edits);
  const launchCommand =
    output &&
    (runMode === "sdk"
      ? output.command.replace(/fastvideo generate --config config\.yaml$/, sdkCommand)
      : output.command);
  const groups = [...new Set(fields.map((field) => field.path.split(".").slice(0, -1).join(".")))];
  function update(path: string, raw: string | undefined, field: Field) {
    setNotice("");
    const next = { ...edits };
    const errors = { ...parseErrors };
    delete errors[path];
    if (raw === undefined) delete next[path];
    else {
      try {
        next[path] = parseField(field, raw);
      } catch {
        errors[path] = "Enter a valid number or JSON value for this field.";
      }
    }
    setEdits(next);
    setParseErrors(errors);
  }
  async function copy(text: string, message = "Copied to clipboard.") {
    try {
      await navigator.clipboard.writeText(text);
      setNotice(message);
    } catch {
      setNotice("Copy failed. Select and copy the text below.");
    }
  }
  function download() {
    if (!output) return;
    const url = URL.createObjectURL(new Blob([output.yaml], { type: "application/yaml" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = "config.yaml";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    setNotice("Downloaded config.yaml.");
  }
  const mlx = catalog.runtime.backend === "mlx";
  const generator = catalog.base_config.generator as { engine?: { quantization?: { transformer_quant?: string } } };
  const nvfp4 = generator?.engine?.quantization?.transformer_quant === "NVFP4";
  const hardwareOptions = (
    mlx
      ? [["apple", "Apple Silicon", "Unified memory"]]
      : [
          ["b200", "B200", "Data center"],
          ["gb200", "GB200", "Data center"],
          ["h100", "H100", "Data center"],
          ["rtx5090", "RTX 5090", "GeForce"],
          ["rtx", "RTX PRO 6000", "Workstation"],
          ["spark", "DGX Spark", "GB10"],
        ]
  ).filter(([id]) => !nvfp4 || id !== "h100");
  const selectedHardware = hardwareOptions.find(([value]) => value === hardware);
  const hardwareLabel = hardware.startsWith("custom:") ? hardware.slice(7) : selectedHardware?.[1];
  const model = modelDirectory.find((model) => model.id === catalog.id.split("/")[0])!;
  const renderField = (field: Field) => (
    <FieldControl
      key={`${reset}-${field.path}`}
      field={field}
      edits={edits}
      update={update}
      failure={parseErrors[field.path] || (field.path === failedPath ? failure : undefined)}
    />
  );
  function chooseHardware(value: string) {
    setHardware(value);
    const url = new URL(window.location.href);
    if (value) url.searchParams.set("hardware", value);
    else url.searchParams.delete("hardware");
    window.history.replaceState(null, "", url);
  }
  const modes = mlx || catalog.runtime.interface === "websocket" ? ["serving"] : ["sdk", "inference", "serving"];
  const modeNote = mlx
    ? "Apple Silicon runs through the native MLX server in this recipe. There is no CUDA CLI or Python SDK launch for this deployment."
    : catalog.runtime.interface === "websocket"
      ? "This deployment uses the native WebSocket server. Only serving is recorded for it."
      : "";
  const advancedGroups = groups.filter(
    (group) => group === "server" || group.includes("parallelism") || group.startsWith("streaming"),
  );
  const standardGroups = groups.filter((group) => group !== "generator.engine" && !advancedGroups.includes(group));
  const groupFields = (group: string) =>
    fields.filter((field) => field.path.split(".").slice(0, -1).join(".") === group);
  return (
    <>
      <ModelHeader
        model={model}
        actions={
          <>
            <Button
              variant="outline"
              disabled={!output}
              onClick={() => output && copy(output.yaml, "Copied config.yaml.")}
            >
              Copy YAML
            </Button>
            <Button disabled={!output} onClick={download}>
              Download YAML
            </Button>
          </>
        }
      />
      {
        <div className="recipe-run-modes option-chips" role="group" aria-label="Run mode">
          {modes.map((mode) => (
            <button
              type="button"
              key={mode}
              aria-pressed={runMode === mode}
              onClick={() => {
                setRunMode(mode);
                setNotice("");
              }}
            >
              {mode === "sdk" ? "Python SDK" : mode === "inference" ? "CLI" : "Serving"}
            </button>
          ))}
        </div>
      }
      {modeNote && <p className="recipe-mode-note">{modeNote}</p>}
      <section className="recipe-command-box" aria-label="Generated configuration">
        <Tabs.Root value={outputTab} onValueChange={setOutputTab}>
          <div className="recipe-command-toolbar">
            <div className="recipe-command-tools">
              <button
                type="button"
                disabled={!output}
                onClick={() =>
                  output && copy(configAndLaunch(output.yaml, launchCommand!), "Copied config.yaml and launch command.")
                }
              >
                Copy config + command
              </button>
              {runMode === "serving" && (
                <button
                  type="button"
                  disabled={!result}
                  aria-expanded={showClient}
                  aria-controls={`client-${catalog.id.replaceAll("/", "-")}`}
                  onClick={() => {
                    setOutputTab("commands");
                    setShowClient(!showClient);
                  }}
                >
                  Show request
                </button>
              )}
              {Object.keys(catalog.env).length > 0 && (
                <button type="button" aria-pressed={showEnv} onClick={() => setShowEnv(!showEnv)}>
                  Show environment
                </button>
              )}
            </div>
            <Tabs.List className="recipe-tabs" aria-label="Output format">
              <Tabs.Trigger value="commands">Commands</Tabs.Trigger>
              <Tabs.Trigger value="yaml">config.yaml</Tabs.Trigger>
            </Tabs.List>
          </div>
          {showEnv && (
            <pre className="recipe-environment" tabIndex={0} aria-label="Environment variables">
              <code>
                {Object.entries(catalog.env)
                  .map(([name, value]) => `${name}=${value}`)
                  .join("\n")}
              </code>
            </pre>
          )}
          {failure && (
            <div role="alert" className="recipe-error">
              <p>{failure}</p>
              <button type="button" onClick={() => setAdvanced(true)}>
                Review settings ↓
              </button>
            </div>
          )}
          <Tabs.Content value="commands" className="recipe-command-content">
            {result && output && (
              <>
                <div className="recipe-code-heading">
                  <span>{runMode === "sdk" ? "Python SDK" : runMode === "inference" ? "Generate" : "Serve"}</span>
                </div>
                <pre
                  tabIndex={0}
                  aria-label={
                    runMode === "sdk" ? "SDK command" : runMode === "inference" ? "Inference command" : "Server command"
                  }
                >
                  <code>{launchCommand}</code>
                </pre>
                <p className="matrix-note">
                  Uses your settings; Copy config + command writes config.yaml first. Run from an installed FastVideo
                  checkout.
                </p>
                {runMode === "serving" && (
                  <details
                    className="recipe-request-code"
                    id={`client-${catalog.id.replaceAll("/", "-")}`}
                    open={showClient}
                    onToggle={(event) => setShowClient(event.currentTarget.open)}
                  >
                    <summary>{result.websocketUrl ? "Health check and WebSocket" : "Example request"}</summary>
                    <button type="button" onClick={() => copy(result.clientCommand, "Copied example request.")}>
                      Copy request command
                    </button>
                    <pre tabIndex={0} aria-label="Client command">
                      <code>{result.clientCommand}</code>
                    </pre>
                    {result.websocketUrl && (
                      <p>
                        WebSocket URL: <code>{result.websocketUrl}</code>
                      </p>
                    )}
                  </details>
                )}
              </>
            )}
          </Tabs.Content>
          <Tabs.Content value="yaml" className="recipe-command-content">
            {output && (
              <pre tabIndex={0} aria-label="Generated YAML">
                <code>{output.yaml}</code>
              </pre>
            )}
          </Tabs.Content>
        </Tabs.Root>
        <p role="status" className="recipe-copy-status">
          {notice}
        </p>
      </section>
      <details className="recipe-install">
        <summary>
          Installation <span>{catalog.runtime.label.replace(/\s*REST/gi, "")}</span>
        </summary>
        {catalog.requirements?.length ? (
          <ul>
            {catalog.requirements.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        ) : null}
        <div className="recipe-guide-links">
          {catalog.runtime.install_url && <a href={guideUrl(catalog.runtime.install_url)}>Installation guide ↗</a>}
          {catalog.guide && <a href={guideUrl(catalog.guide.url)}>Model guide ↗</a>}
          {catalog.runtime.client_guide_url && <a href={guideUrl(catalog.runtime.client_guide_url)}>Client guide ↗</a>}
        </div>
      </details>
      <section className="recipe-matrix" aria-label="Configuration settings">
        <MatrixRow title="Hardware">
          <div className="option-chips hardware-backends" role="group" aria-label="Hardware backend">
            {deployments.map((deployment) => (
              <button
                key={deployment.id}
                type="button"
                aria-pressed={catalog.id === deployment.id}
                onClick={() => {
                  if (catalog.id !== deployment.id) selectRuntime(deployment.id);
                }}
              >
                {deployment.runtime.includes("mlx") ? "MLX" : "CUDA"}
              </button>
            ))}
          </div>
          <div className="hardware-resource-grid">
            <HardwarePicker
              backend={mlx ? "mlx" : "nvidia"}
              options={hardwareOptions}
              value={hardware}
              onChange={chooseHardware}
            />
            <div className="matrix-resources">
              <span className="matrix-resource-title">Resources</span>
              {fields.filter((field) => field.path === "generator.engine.num_gpus").map(renderField)}
              {mlx && <p>1 Apple Silicon host</p>}
              <a
                href={sitePath(
                  mlx
                    ? "/getting_started/installation/mlx"
                    : hardware === "spark"
                      ? "/getting_started/installation/spark"
                      : "/getting_started/installation/gpu",
                )}
              >
                Setup guide ↗
              </a>
            </div>
          </div>
          {hardwareLabel ? (
            <p className="matrix-note">
              {mlx
                ? `${hardwareLabel} is saved as your target. Apple Silicon uses unified memory; dedicated GPU memory does not apply.`
                : `${hardwareLabel} is saved as your target. It does not change the generated YAML; use GPU count and parallelism for the run.`}
            </p>
          ) : null}
          <RecipeFacts validity={failure ? "invalid" : result ? "valid" : "loading"} fit="unknown" evidence="none" />
        </MatrixRow>
        {catalog.workload === "i2v" && runMode !== "serving" && (
          <MatrixRow title="Inputs">
            <div className="matrix-field">
              <label htmlFor={sourceImageId} className="matrix-field-label">
                Source image path
              </label>
              <Input
                id={sourceImageId}
                value={sourceImage}
                required
                aria-invalid={!sourceImage.trim()}
                aria-describedby={`${sourceImageId}-hint`}
                placeholder="/path/to/first-frame.png"
                onChange={(event) => {
                  setSourceImage(event.target.value);
                  setNotice("");
                }}
              />
              <p id={`${sourceImageId}-hint`} className="matrix-note">
                Path to an image on the machine running FastVideo. Included as request.inputs.image_path.
              </p>
            </div>
          </MatrixRow>
        )}
        {model.family === "FastH3" && (
          <MatrixRow title="Components">
            {model.id === "fasth3-trim" && (
              <div className="option-chips" aria-label="Recipe components">
                {["NVFP4 transformer", "50-layer NVFP4 encoder", "Lightweight video VAE", "H3 audio VAE"].map(
                  (name) => (
                    <span className="option-chip-static" key={name}>
                      {name}
                    </span>
                  ),
                )}
              </div>
            )}
            <p className="matrix-note">
              Components come from the maintained recipe. They are not individually selectable in the cookbook yet.
            </p>
          </MatrixRow>
        )}
        {standardGroups.map((group) => (
          <MatrixRow title={groupTitle(group) ?? group} key={group}>
            <div className="matrix-options-grid">{groupFields(group).map(renderField)}</div>
          </MatrixRow>
        ))}
        <details
          className="matrix-advanced"
          open={advanced}
          onToggle={(event) => setAdvanced(event.currentTarget.open)}
        >
          <summary>
            Advanced <span>Server, topology, and runtime settings</span>
          </summary>
          {advancedGroups.map((group) => (
            <MatrixRow title={groupTitle(group) ?? group} key={group}>
              <div className="matrix-options-grid">{groupFields(group).map(renderField)}</div>
              {group.includes("parallelism") && (
                <button
                  className="matrix-auto"
                  type="button"
                  onClick={() => {
                    const next = { ...edits };
                    for (const field of fields.filter((field) => /parallelism\.(tp_size|sp_size)$/.test(field.path)))
                      next[field.path] = -1;
                    setEdits(next);
                    setReset(reset + 1);
                    setParseErrors(
                      Object.fromEntries(Object.entries(parseErrors).filter(([path]) => !path.includes("parallelism"))),
                    );
                    setNotice("Parallelism set to automatic.");
                  }}
                >
                  Use automatic parallelism
                </button>
              )}
            </MatrixRow>
          ))}
        </details>
      </section>
      <div className="recipe-bottom-actions">
        <span>Copy config and launch together, or download config.yaml first.</span>
        <Button
          variant="ghost"
          onClick={() => {
            setEdits({});
            setSourceImage("");
            setParseErrors({});
            setReset(reset + 1);
            setNotice("Restored the baseline configuration.");
          }}
        >
          Reset changes
        </Button>
      </div>
    </>
  );
}

function MatrixRow({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="matrix-row">
      <h2>{title}</h2>
      <div className="matrix-row-content">{children}</div>
    </div>
  );
}

export default function Cookbook({ modelId }: { modelId?: string }) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  const [index, setIndex] = useState<Index>();
  const [selected, setSelected] = useState("");
  const [catalog, setCatalog] = useState<Catalog>();
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    setError("");
    const url = new URL(
      `${import.meta.env.BASE_URL.replace(/\/$/, "")}/assets/cookbook-config/index.json`,
      window.location.origin,
    );
    api
      .loadIndex(url.href, { signal: controller.signal })
      .then((data) => {
        if (!data.models.some((model) => model.deployments.length)) {
          throw new Error("No cookbook deployments are available. Check the catalog and retry.");
        }
        setIndex(data);
        const requested = new URL(window.location.href).searchParams.get("deployment");
        const requestedModel = data.models.find((model) =>
          model.deployments.some((deployment) => deployment.id === requested),
        );
        if (!modelId && requestedModel) {
          const page = recipePages.find((page) => page.id === requestedModel.id);
          if (page) {
            window.location.replace(`${sitePath(page.path)}${window.location.search}`);
            return;
          }
        }
        const deployments = data.models
          .filter((model) => !modelId || model.id === modelId)
          .flatMap((model) => model.deployments);
        setSelected(deployments.some((item) => item.id === requested) ? requested! : (deployments[0]?.id ?? ""));
      })
      .catch((error) => {
        if (!controller.signal.aborted) setError(errorText(error));
      });
    return () => controller.abort();
  }, [retry, modelId]);
  useEffect(() => {
    if (!index || !selected || !modelId || !recipePages.some((page) => page.id === modelId)) return;
    const controller = new AbortController();
    setCatalog(undefined);
    setError("");
    api
      .loadDeployment(index, selected, { signal: controller.signal })
      .then((data) => {
        if (!controller.signal.aborted) setCatalog(data);
      })
      .catch((error) => {
        if (!controller.signal.aborted) setError(errorText(error));
      });
    return () => controller.abort();
  }, [index, selected, modelId]);
  function select(value: string) {
    setSelected(value);
    setCatalog(undefined);
    setError("");
    const url = new URL(window.location.href);
    url.searchParams.set("deployment", value);
    window.history.replaceState(null, "", url);
  }
  const model = index?.models.find((model) => model.id === modelId);
  if (!modelId) return <RecipeDirectory index={index} error={error} retry={() => setRetry(retry + 1)} />;
  const directoryModel = modelDirectory.find((model) => model.id === modelId);
  if (directoryModel && !directoryModel.configurable)
    return mounted ? (
      <SourceRecipe model={directoryModel} />
    ) : (
      <div className="cookbook not-prose recipe-detail">
        <ModelHeader model={directoryModel} />
        <p>{directoryModel.description}</p>
        <p className="recipe-loading">Loading recipe…</p>
        <a href={`https://github.com/hao-ai-lab/FastVideo/blob/main/${directoryModel.recipes[0].source}`}>
          View maintained example ↗
        </a>
      </div>
    );
  return (
    <div className="cookbook not-prose recipe-detail">
      {error ? (
        <div role="alert" className="space-y-3 rounded-lg border p-4">
          <p>{error}</p>
          <Button
            variant="outline"
            onClick={() => {
              setCatalog(undefined);
              setIndex(undefined);
              setRetry(retry + 1);
            }}
          >
            Retry loading recipes
          </Button>
        </div>
      ) : catalog?.id === selected ? (
        <Editor key={catalog.id} catalog={catalog} deployments={model?.deployments ?? []} selectRuntime={select} />
      ) : (
        <div className="recipe-loading">
          {directoryModel && <ModelHeader model={directoryModel} />}
          <div className="option-chips" role="group" aria-label="Hardware backend">
            {model?.deployments.map((deployment) => (
              <button
                key={deployment.id}
                type="button"
                aria-pressed={selected === deployment.id}
                onClick={() => select(deployment.id)}
              >
                {deployment.runtime.includes("mlx") ? "MLX" : "CUDA"}
              </button>
            ))}
          </div>
          <p role="status">Loading deployment settings…</p>
        </div>
      )}
    </div>
  );
}

function RecipeDirectory({ index, error, retry }: { index?: Index; error: string; retry: () => void }) {
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("All models");
  const [family, setFamily] = useState("All families");
  const visible = modelDirectory.filter(
    (model) =>
      (category === "All models" || model.category === category) &&
      (family === "All families" || model.family === family) &&
      `${model.title} ${model.family} ${model.tags.join(" ")}`.toLowerCase().includes(query.trim().toLowerCase()),
  );
  return (
    <div className="cookbook not-prose recipe-directory">
      <div className="recipe-directory-tools">
        <div className="recipe-family-filters" aria-label="Filter recipes by category">
          {["All models", "Video", "Image", "Audio", "World"].map((value) => (
            <button
              key={value}
              type="button"
              aria-pressed={category === value}
              onClick={() => {
                setCategory(value);
                setFamily("All families");
              }}
            >
              {value}
            </button>
          ))}
        </div>
        <Input
          aria-label="Search recipes"
          type="search"
          placeholder="Search models…"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
        />
      </div>
      <div className="recipe-directory-subbar">
        <p role="status">{visible.length} models</p>
        <select aria-label="Model family" value={family} onChange={(event) => setFamily(event.target.value)}>
          <option>All families</option>
          {[
            ...new Set(
              modelDirectory
                .filter((model) => category === "All models" || model.category === category)
                .map((model) => model.family),
            ),
          ].map((family) => (
            <option key={family}>{family}</option>
          ))}
        </select>
      </div>
      {error && (
        <div role="alert" className="recipe-error">
          <p>{error}</p>
          <Button variant="outline" onClick={retry}>
            Retry loading recipes
          </Button>
        </div>
      )}
      <div className="recipe-card-grid">
        {visible.map((model) => (
          <a className="recipe-card" key={model.id} href={sitePath(model.path)}>
            <div className="recipe-card-top">
              <img src={sitePath(`/assets/logos/${model.logo}.webp`)} alt="" width="40" height="40" />
              <span className="recipe-card-kicker">{model.provider}</span>
              {model.featured && <span className="recipe-featured">Featured</span>}
            </div>
            <h2>{model.title}</h2>
            <div className="recipe-tags">
              <span>{model.category}</span>
              {model.tags.slice(0, 2).map((tag) => (
                <span key={tag}>{tag}</span>
              ))}
            </div>
            <footer>
              <span>
                {model.platforms.map((platform) => (platform === "Apple Silicon" ? "MLX" : platform)).join(" · ")}
              </span>
              <span>{model.configurable ? "Configure →" : "View recipe →"}</span>
            </footer>
          </a>
        ))}
      </div>
      {!visible.length && (
        <div className="recipe-empty">
          <h2>No matching recipes</h2>
          <Button
            variant="outline"
            onClick={() => {
              setQuery("");
              setCategory("All models");
              setFamily("All families");
            }}
          >
            Clear filters
          </Button>
        </div>
      )}
    </div>
  );
}
