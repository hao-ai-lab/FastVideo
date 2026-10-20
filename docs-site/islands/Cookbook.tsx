import { useEffect, useId, useRef, useState, type ReactNode } from "react";
import { MotionConfig } from "motion/react";
import { ArrowUpRight } from "lucide-react";
import { Tabs } from "radix-ui";
import { modelDirectory, recipePages, sitePath, type DirectoryModel } from "@/lib/cookbook-pages";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import SourceRecipe from "./SourceRecipe";
import HardwarePicker from "@/components/cookbook/HardwarePicker";
import ModelHeader from "@/components/cookbook/ModelHeader";
import {
  CardCurtain,
  CardCurtainReveal,
  CardCurtainRevealBody,
  CardCurtainRevealDescription,
  CardCurtainRevealFooter,
  CardCurtainRevealTitle,
} from "@/components/ui/card-curtain-reveal";
import RecipeFacts from "@/components/cookbook/RecipeFacts";
import { configAndLaunch, inferenceOutput, sdkCommand } from "@/lib/cookbook-commands";
import {
  gpuCountFrom,
  readHardwareSelection,
  resolveHardware,
  useHardwareCatalog,
  writeHardwareSelection,
  type HardwareSelection,
} from "@/lib/hardware";
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
    "generator.pipeline.experimental": "Decoder",
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
  const [hardware, setHardware] = useState<HardwareSelection>(() =>
    readHardwareSelection(new URL(window.location.href).searchParams),
  );
  const hardwareCatalog = useHardwareCatalog();
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
    let next = { ...edits };
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
    if (path === GPU_COUNT_PATH && typeof next[path] === "number") {
      const fitted = fitParallelism(fields, next, next[path] as number);
      if (fitted !== next) {
        next = fitted;
        for (const key of Object.keys(errors)) if (key.includes("parallelism")) delete errors[key];
        setReset(reset + 1);
        setNotice(`Parallelism set to automatic for ${next[path]} ${next[path] === 1 ? "GPU" : "GPUs"}.`);
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
  const hardwareRecord = resolveHardware(hardware, hardwareCatalog.records);
  const gpuCount = gpuCountFrom(result?.config);
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
  function chooseHardware(selection: HardwareSelection) {
    setHardware(selection);
    const url = new URL(window.location.href);
    writeHardwareSelection(url, selection);
    window.history.pushState(null, "", url);
    const record = resolveHardware(selection, hardwareCatalog.records);
    const countField = fields.find((field) => field.path === GPU_COUNT_PATH);
    if (record && countField && singleGpuDevice(record) && gpuCount > 1) {
      setEdits(fitParallelism(fields, { ...edits, [GPU_COUNT_PATH]: 1 }, 1));
      setParseErrors(Object.fromEntries(Object.entries(parseErrors).filter(([path]) => !path.includes("parallelism"))));
      setReset(reset + 1);
      setNotice(`Set to 1 GPU for ${record.name}. Raise GPU count if this machine has more.`);
    }
  }
  useEffect(() => {
    const restore = () => setHardware(readHardwareSelection(new URL(window.location.href).searchParams));
    window.addEventListener("popstate", restore);
    return () => window.removeEventListener("popstate", restore);
  }, []);
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
              backend={mlx ? "mlx" : "cuda"}
              catalog={hardwareCatalog.records}
              status={hardwareCatalog.status}
              error={hardwareCatalog.error}
              reload={hardwareCatalog.reload}
              value={hardware}
              onChange={chooseHardware}
              gpuCount={gpuCount}
            />
            <div className="matrix-resources">
              <span className="matrix-resource-title">Resources</span>
              {fields.filter((field) => field.path === "generator.engine.num_gpus").map(renderField)}
              {mlx && <p>1 Apple Silicon host</p>}
              <a
                href={sitePath(
                  mlx
                    ? "/getting_started/installation/mlx"
                    : hardwareRecord?.name.toLowerCase().includes("dgx spark")
                      ? "/getting_started/installation/spark"
                      : "/getting_started/installation/gpu",
                )}
              >
                Setup guide ↗
              </a>
            </div>
          </div>
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
        {model.family === "MiniMax H3" && (
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
              Weights and encoders come from the maintained recipe. Choose the video decoder under Decoder.
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
    window.history.pushState(null, "", url);
  }
  useEffect(() => {
    const restore = () => {
      const requested = new URL(window.location.href).searchParams.get("deployment");
      const available = index?.models.find((item) => item.id === modelId)?.deployments ?? [];
      const next = available.find((deployment) => deployment.id === requested)?.id ?? available[0]?.id;
      if (!next) return;
      setSelected(next);
      setCatalog(undefined);
      setError("");
    };
    window.addEventListener("popstate", restore);
    return () => window.removeEventListener("popstate", restore);
  }, [index, modelId]);
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

const GPU_COUNT_PATH = "generator.engine.num_gpus";
const SINGLE_GPU_FORM_FACTOR = /desktop|laptop|workstation|system|soc/;

/** Desktop and workstation cards, DGX Spark, and Macs are one-GPU machines; datacenter parts keep the recipe default. */
function singleGpuDevice(record: { form_factor?: string }) {
  return SINGLE_GPU_FORM_FACTOR.test(record.form_factor ?? "");
}

/** Returns new edits where every fixed parallel size that no longer divides `gpus` becomes automatic. */
function fitParallelism(fields: Field[], edits: Record<string, Value>, gpus: number): Record<string, Value> {
  const stale = fields.filter((field) => {
    if (!/\.parallelism\.[a-z_]+_size$/.test(field.path)) return false;
    const size = edits[field.path] ?? field.value;
    return typeof size === "number" && size > 0 && (size > gpus || gpus % size !== 0);
  });
  return stale.length ? { ...edits, ...Object.fromEntries(stale.map((field) => [field.path, -1])) } : edits;
}

/** Mirrors the editor: MLX and WebSocket deployments offer serving only. */
function configurableModes(index: Index | undefined, modelId: string): string[] | undefined {
  const deployments = index?.models.find((model) => model.id === modelId)?.deployments;
  if (!deployments?.length) return undefined;
  const servingOnly = (runtime: string) => runtime.includes("mlx") || runtime.includes("streaming");
  return deployments.some((deployment) => !servingOnly(deployment.runtime))
    ? ["Python SDK", "CLI", "Serving"]
    : ["Serving"];
}

/** Mirrors the source recipe page: Python SDK and CLI always, serving when a recipe declares it. */
function exampleModes(model: DirectoryModel): string[] {
  return ["Python SDK", "CLI", ...(model.recipes.some((recipe) => recipe.serving) ? ["Serving"] : [])];
}

/** Rest state shows identity only; hover or keyboard focus opens a curtain over the details. */
function RecipeCard({ model, href, modes }: { model: DirectoryModel; href: string; modes?: string[] }) {
  const [focused, setFocused] = useState(false);
  const details = useRef<HTMLDivElement>(null);
  const [restOffset, setRestOffset] = useState(72);
  useEffect(() => {
    const element = details.current;
    if (!element) return;
    const measure = () => setRestOffset(element.offsetHeight + 8);
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  return (
    <a className="recipe-card" href={href} onFocus={() => setFocused(true)} onBlur={() => setFocused(false)}>
      <CardCurtainReveal active={focused} className="recipe-card-reveal">
        <CardCurtainRevealBody className="recipe-card-body">
          <div className="recipe-card-top">
            <img src={sitePath(`/assets/logos/${model.logo}.webp`)} alt="" width="40" height="40" />
            <span className="recipe-card-kicker">{model.provider}</span>
            {model.featured ? <span className="recipe-featured">Featured</span> : null}
          </div>
          <CardCurtainRevealTitle
            as={model.configurable ? "h3" : "h4"}
            hiddenOffset={restOffset}
            className="recipe-card-title"
          >
            {model.title}
          </CardCurtainRevealTitle>
          <CardCurtainRevealDescription ref={details} className="recipe-card-details">
            <p className="recipe-card-description">{model.description}</p>
            <div className="recipe-tags">
              <span>{model.category}</span>
              {model.tags.slice(0, 2).map((tag) => (
                <span key={tag}>{tag}</span>
              ))}
            </div>
            {modes ? <p className="recipe-card-modes">{modes.join(" · ")}</p> : null}
          </CardCurtainRevealDescription>
          <CardCurtain className="recipe-card-curtain" />
        </CardCurtainRevealBody>
        <CardCurtainRevealFooter className="recipe-card-footer">
          <span>
            {model.platforms.map((platform) => (platform === "Apple Silicon" ? "MLX" : platform)).join(" · ")}
          </span>
          <span className="recipe-card-cta">
            {model.configurable ? "Configure" : "View recipe"} <ArrowUpRight aria-hidden="true" />
          </span>
        </CardCurtainRevealFooter>
      </CardCurtainReveal>
    </a>
  );
}

function RecipeDirectory({ index, error, retry }: { index?: Index; error: string; retry: () => void }) {
  const [initial] = useState(() =>
    typeof window === "undefined" ? new URLSearchParams() : new URL(window.location.href).searchParams,
  );
  const [query, setQuery] = useState(initial.get("q") ?? "");
  const [category, setCategory] = useState(initial.get("category") ?? "All models");
  const [family, setFamily] = useState(initial.get("family") ?? "All families");
  const [runtime, setRuntime] = useState(initial.get("runtime") ?? "all");
  const [task, setTask] = useState(initial.get("task") ?? "All tasks");
  useEffect(() => {
    const url = new URL(window.location.href);
    const search = url.searchParams;
    const set = (key: string, value: string | undefined) => {
      if (value) search.set(key, value);
      else search.delete(key);
    };
    set("q", query.trim() || undefined);
    set("category", category === "All models" ? undefined : category);
    set("family", family === "All families" ? undefined : family);
    set("runtime", runtime === "all" ? undefined : runtime);
    set("task", task === "All tasks" ? undefined : task);
    window.history.replaceState(null, "", url);
  }, [category, family, query, runtime, task]);
  const runtimeMatches = (platforms: string[]) => {
    if (runtime === "cuda") return platforms.includes("NVIDIA");
    if (runtime === "mlx") return platforms.includes("Apple Silicon");
    return true;
  };
  const visible = modelDirectory.filter(
    (model) =>
      (category === "All models" || model.category === category) &&
      (family === "All families" || model.family === family) &&
      (task === "All tasks" || model.task === task) &&
      runtimeMatches(model.platforms) &&
      `${model.title} ${model.family} ${model.tags.join(" ")} ${model.description}`
        .toLowerCase()
        .includes(query.trim().toLowerCase()),
  );
  function clear() {
    setQuery("");
    setCategory("All models");
    setFamily("All families");
    setRuntime("all");
    setTask("All tasks");
  }
  const configurable = visible.filter((model) => model.configurable);
  const examples = visible.filter((model) => !model.configurable);
  const exampleFamilies = [...new Set(examples.map((model) => model.family))];
  const filtered =
    !!query.trim() ||
    category !== "All models" ||
    family !== "All families" ||
    runtime !== "all" ||
    task !== "All tasks";
  const categoryCount = (value: string) =>
    modelDirectory.filter((model) => value === "All models" || model.category === value).length;
  return (
    <MotionConfig reducedMotion="user">
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
                <span className="recipe-filter-count" aria-hidden="true">
                  {categoryCount(value)}
                </span>
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
          <p role="status">
            {visible.length} {visible.length === 1 ? "model" : "models"}
            {configurable.length ? ` · ${configurable.length} configurable` : ""}
            {runtime !== "all" ? ` · ${runtime === "mlx" ? "MLX" : "CUDA"} only` : ""}
            {task !== "All tasks" ? ` · ${task}` : ""}
            {filtered && visible.length ? (
              <button type="button" className="recipe-clear" onClick={clear}>
                Reset filters
              </button>
            ) : null}
          </p>
          <div className="recipe-directory-selects">
            <label>
              <span>Family</span>
              <select aria-label="Model family" value={family} onChange={(event) => setFamily(event.target.value)}>
                <option>All families</option>
                {[
                  ...new Set(
                    modelDirectory
                      .filter((model) => category === "All models" || model.category === category)
                      .map((model) => model.family),
                  ),
                ].map((value) => (
                  <option key={value}>{value}</option>
                ))}
              </select>
            </label>
            <label>
              <span>Task</span>
              <select aria-label="Task" value={task} onChange={(event) => setTask(event.target.value)}>
                <option>All tasks</option>
                {[...new Set(modelDirectory.map((model) => model.task))].sort().map((value) => (
                  <option key={value}>{value}</option>
                ))}
              </select>
            </label>
            <label>
              <span>Runtime</span>
              <select aria-label="Runtime" value={runtime} onChange={(event) => setRuntime(event.target.value)}>
                <option value="all">All runtimes</option>
                <option value="cuda">CUDA</option>
                <option value="mlx">MLX</option>
              </select>
            </label>
          </div>
        </div>
        {error ? (
          <div role="alert" className="recipe-error">
            <p>{error}</p>
            <Button variant="outline" onClick={retry}>
              Retry loading recipes
            </Button>
          </div>
        ) : null}
        {configurable.length ? (
          <section className="recipe-section" aria-labelledby="recipe-configurable-heading">
            <header className="recipe-section-head">
              <h2 id="recipe-configurable-heading">Configurable recipes</h2>
              <p>Edit settings, then copy one config for the Python SDK, the CLI, or an OpenAI-compatible server.</p>
            </header>
            <div className="recipe-card-grid recipe-card-grid-configurable">
              {configurable.map((model) => (
                <RecipeCard
                  key={model.id}
                  model={model}
                  href={sitePath(model.path)}
                  modes={configurableModes(index, model.id)}
                />
              ))}
            </div>
          </section>
        ) : null}
        {examples.length ? (
          <section className="recipe-section" aria-labelledby="recipe-examples-heading">
            <header className="recipe-section-head">
              <h2 id="recipe-examples-heading">Example recipes</h2>
              <p>
                Maintained example scripts, grouped by model family. Adjust their documented settings and run them as
                written.
              </p>
            </header>
            {exampleFamilies.map((name) => {
              const members = examples.filter((model) => model.family === name);
              return (
                <div className="recipe-family-group" key={name}>
                  <div className="recipe-family-head">
                    <h3>{name}</h3>
                    <span>
                      {members.length} {members.length === 1 ? "recipe" : "recipes"}
                    </span>
                    <a href={sitePath(members[0].familyPath)}>Family guide →</a>
                  </div>
                  <div className="recipe-card-grid">
                    {members.map((model) => (
                      <RecipeCard
                        key={model.id}
                        model={model}
                        href={sitePath(model.path)}
                        modes={exampleModes(model)}
                      />
                    ))}
                  </div>
                </div>
              );
            })}
          </section>
        ) : null}
        {!visible.length ? (
          <div className="recipe-empty">
            <h2>No matching recipes</h2>
            <p>Clear a filter or choose a different runtime.</p>
            <Button variant="outline" onClick={clear}>
              Clear filters
            </Button>
          </div>
        ) : null}
      </div>
    </MotionConfig>
  );
}
