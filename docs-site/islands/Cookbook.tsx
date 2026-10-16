import { useEffect, useId, useState, type ReactNode } from "react";
import { Tabs } from "radix-ui";
import { modelDirectory, recipePages, sitePath, type DirectoryModel } from "@/lib/cookbook-pages";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import SourceRecipe from "./SourceRecipe";
import ModelHeader from "@/components/cookbook/ModelHeader";
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
  const [hardware, setHardware] = useState(
    () => new URL(window.location.href).searchParams.get("hardware") ?? "unspecified",
  );
  const [edits, setEdits] = useState<Record<string, Value>>({});
  const [parseErrors, setParseErrors] = useState<Record<string, string>>({});
  const [reset, setReset] = useState(0);
  const [notice, setNotice] = useState("");
  let result: Result | undefined;
  let failure = "";
  let failedPath: string | undefined;
  try {
    if (Object.keys(parseErrors).length) throw new Error("Correct the highlighted fields to generate a configuration.");
    result = api.resolveConfig(catalog, edits);
  } catch (error) {
    failure = errorText(error);
    failedPath = (error as { path?: string }).path;
  }
  const fields = api.getOptions(catalog, catalog.base_config, edits);
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
  async function copy(text: string) {
    try {
      await navigator.clipboard.writeText(text);
      setNotice("Copied to clipboard.");
    } catch {
      setNotice("Copy failed. Select and copy the text below.");
    }
  }
  function download() {
    if (!result) return;
    const url = URL.createObjectURL(new Blob([result.yaml], { type: "application/yaml" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = "config.yaml";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    setNotice("Downloaded config.yaml.");
  }
  const mlx = catalog.runtime.backend === "mlx";
  const generator = catalog.base_config.generator as Record<string, Value>;
  const engine = generator.engine as Record<string, Value> | undefined;
  const quantization = engine?.quantization as Record<string, Value> | undefined;
  const modelPath = String(generator.model_path);
  const hardwareOptions = mlx
    ? [["apple", "Apple Silicon", "Unified memory"]]
    : [
        ["b200", "B200", "Data center"],
        ["gb200", "GB200", "Data center"],
        ["h100", "H100", "Data center"],
        ["rtx5090", "RTX 5090", "GeForce"],
        ["rtx", "RTX PRO 6000", "Workstation"],
        ["spark", "DGX Spark", "GB10"],
      ];
  const selectedHardware = hardwareOptions.find(([value]) => value === hardware) ?? hardwareOptions[0];
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
    url.searchParams.set("hardware", value);
    window.history.replaceState(null, "", url);
  }
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
            <Button variant="outline" disabled={!result} onClick={() => result && copy(result.yaml)}>
              Copy YAML
            </Button>
            <Button disabled={!result} onClick={download}>
              Download YAML
            </Button>
          </>
        }
      />
      <section className="recipe-command-box" aria-label="Generated configuration">
        <Tabs.Root defaultValue="commands">
          <div className="recipe-command-toolbar">
            <span className="recipe-unverified">Unverified recipe</span>
            <Tabs.List className="recipe-tabs" aria-label="Output format">
              <Tabs.Trigger value="commands">Commands</Tabs.Trigger>
              <Tabs.Trigger value="yaml">config.yaml</Tabs.Trigger>
            </Tabs.List>
          </div>
          {failure && (
            <div role="alert" className="recipe-error">
              <p>{failure}</p>
              <button type="button" onClick={() => setAdvanced(true)}>
                Review settings ↓
              </button>
            </div>
          )}
          <Tabs.Content value="commands" className="recipe-command-content">
            {result && (
              <>
                <div className="recipe-code-heading">
                  <span>Serve</span>
                  <button type="button" onClick={() => copy(result.command)}>
                    Copy server command
                  </button>
                </div>
                <pre tabIndex={0} aria-label="Server command">
                  <code>{result.command}</code>
                </pre>
                <details className="recipe-request-code">
                  <summary>{result.websocketUrl ? "Health check and WebSocket" : "Example request"}</summary>
                  <button type="button" onClick={() => copy(result.clientCommand)}>
                    Copy client command
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
              </>
            )}
          </Tabs.Content>
          <Tabs.Content value="yaml" className="recipe-command-content">
            {result && (
              <pre tabIndex={0} aria-label="Generated YAML">
                <code>{result.yaml}</code>
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
          Installation <span>{catalog.runtime.label}</span>
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
        <MatrixRow title="Base model">
          <select
            aria-label="Base model"
            value={model.id}
            onChange={(event) => {
              const selected = modelDirectory.find((model) => model.id === event.target.value);
              if (selected) window.location.assign(sitePath(selected.path));
            }}
          >
            {modelDirectory
              .filter((entry) => entry.family === model.family)
              .map((entry) => (
                <option key={entry.id} value={entry.id}>
                  {entry.title}
                </option>
              ))}
          </select>
          <span className="matrix-baseline">
            {quantization?.transformer_quant
              ? `Transformer: ${String(quantization.transformer_quant)}`
              : "Recipe checkpoint"}
          </span>
        </MatrixRow>
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
                {deployment.runtime.includes("mlx") ? "MLX" : "NVIDIA"}
              </button>
            ))}
          </div>
          <div className="hardware-resource-grid">
            <fieldset className="hardware-matrix">
              <legend className="sr-only">Hardware target</legend>
              {hardwareOptions.map(([value, title, detail]) => (
                <label key={value} data-selected={selectedHardware[0] === value}>
                  <input
                    type="radio"
                    name={`hardware-${catalog.id}`}
                    checked={selectedHardware[0] === value}
                    onChange={() => chooseHardware(value)}
                    value={value}
                  />
                  <span>
                    <strong>{title}</strong>
                    <small>{detail}</small>
                  </span>
                </label>
              ))}
            </fieldset>
            <div className="matrix-resources">
              <span className="matrix-resource-title">Resources</span>
              {fields.filter((field) => field.path === "generator.engine.num_gpus").map(renderField)}
              {mlx && <p>1 Apple Silicon host</p>}
              <a
                href={sitePath(
                  mlx
                    ? "/getting_started/installation/mlx"
                    : selectedHardware[0] === "spark"
                      ? "/getting_started/installation/spark"
                      : "/getting_started/installation/gpu",
                )}
              >
                Setup guide ↗
              </a>
            </div>
          </div>
          <p className="matrix-note">Hardware fit has not been verified. GPU count updates the YAML.</p>
        </MatrixRow>
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
        <span>Save config.yaml before running the server.</span>
        <Button
          variant="ghost"
          onClick={() => {
            setEdits({});
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
  if (directoryModel && !directoryModel.configurable) return <SourceRecipe model={directoryModel} />;
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
                {deployment.runtime.includes("mlx") ? "MLX" : "NVIDIA"}
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
              <span className="recipe-card-kicker">{model.family}</span>
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
