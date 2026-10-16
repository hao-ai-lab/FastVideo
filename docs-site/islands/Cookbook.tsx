import { useEffect, useId, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
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
    "generator.engine": "Hardware",
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
    <div className="space-y-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <label htmlFor={id} className="text-sm font-medium">
          {label}
        </label>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          aria-label={`Use baseline for ${label}`}
          onClick={() => {
            setRaw(undefined);
            update(field.path, undefined, field);
          }}
        >
          Use baseline
        </Button>
      </div>
      {choices ? (
        <select
          {...props}
          value={value === undefined ? "inherit" : JSON.stringify(value)}
          onChange={(e) => update(field.path, e.target.value === "inherit" ? undefined : e.target.value, field)}
        >
          <option value="inherit">
            Use baseline
            {field.value === undefined ? " (inherited)" : ` (${String(field.value)})`}
          </option>
          {choices.map((choice) => (
            <option key={JSON.stringify(choice)} value={JSON.stringify(choice)}>
              {String(choice)}
            </option>
          ))}
          {nullable && !choices.includes(null) && <option value="null">null</option>}
        </select>
      ) : kind.includes("object") || kind.includes("array") ? (
        <Textarea
          {...props}
          value={displayed}
          onChange={(e) => edit(e.target.value)}
          placeholder="Inherited, enter JSON to override"
        />
      ) : (
        <Input
          {...props}
          type={kind.includes("integer") || kind.includes("number") ? "number" : "text"}
          value={displayed}
          placeholder={value === undefined ? "Inherited" : value === null ? "null" : ""}
          min={field.schema.minimum}
          max={field.schema.maximum}
          step={field.schema.multipleOf ?? (kind.includes("integer") ? 1 : "any")}
          onChange={(e) => edit(e.target.value)}
        />
      )}
      {nullable && !choices && (
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={() => {
            setRaw(undefined);
            update(field.path, "null", {
              ...field,
              schema: { ...field.schema, enum: [null] },
            });
          }}
        >
          Set {label} to null
        </Button>
      )}
      <p id={`${id}-hint`} className="text-xs text-muted-foreground">
        {field.schema.description || field.path}
        {value === undefined ? " · Inherited from the runtime" : value === null ? " · Explicit null" : ""}
      </p>
      {failure && (
        <p id={`${id}-error`} className="text-sm" role="alert">
          {failure}
        </p>
      )}
    </div>
  );
}

function Editor({ catalog }: { catalog: Catalog }) {
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
  return (
    <>
      <div className="flex flex-wrap items-center gap-2">
        <Badge variant="secondary">{catalog.runtime.label}</Badge>
        <Badge variant="outline">VRAM fit unknown</Badge>
      </div>
      <p className="text-sm text-muted-foreground">{catalog.summary}</p>
      <div className="grid min-w-0 gap-8 xl:grid-cols-2">
        <section aria-label="Configuration settings" className="min-w-0 space-y-5">
          <div className="flex items-center justify-between gap-3">
            <h2 className="text-lg font-semibold">Settings</h2>
            <Button
              variant="outline"
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
          {groups.map((group) => (
            <fieldset key={group} className="min-w-0 space-y-4 rounded-lg border p-4">
              <legend className="px-2 text-sm font-medium">{groupTitle(group)}</legend>
              {fields
                .filter(
                  (field) =>
                    field.path.startsWith(`${group}.`) && field.path.split(".").slice(0, -1).join(".") === group,
                )
                .map((field) => (
                  <FieldControl
                    key={`${reset}-${field.path}`}
                    field={field}
                    edits={edits}
                    update={update}
                    failure={parseErrors[field.path] || (field.path === failedPath ? failure : undefined)}
                  />
                ))}
            </fieldset>
          ))}
        </section>
        <section aria-label="Generated configuration" className="min-w-0 space-y-4">
          <h2 className="text-lg font-semibold">Run this deployment</h2>
          {catalog.requirements?.length ? (
            <ul className="list-disc space-y-2 ps-5 text-sm">
              {catalog.requirements.map((note) => (
                <li key={note}>{note}</li>
              ))}
            </ul>
          ) : null}
          <div className="flex flex-wrap gap-4 text-sm underline">
            {catalog.runtime.install_url && <a href={catalog.runtime.install_url}>Installation guide</a>}
            {catalog.guide && <a href={guideUrl(catalog.guide.url)}>Deployment guide</a>}
            {catalog.runtime.client_guide_url && <a href={guideUrl(catalog.runtime.client_guide_url)}>Client guide</a>}
          </div>
          <p className="text-sm text-muted-foreground">
            Save config.yaml, then run these commands on your host. Hardware capacity and runtime compatibility need
            separate verification.
          </p>
          {failure && (
            <p role="alert" className="rounded-lg border p-4 text-sm">
              {failure}
            </p>
          )}
          <div className="flex flex-wrap gap-2">
            <Button disabled={!result} onClick={download}>
              Download YAML
            </Button>
            <Button variant="outline" disabled={!result} onClick={() => result && copy(result.yaml)}>
              Copy YAML
            </Button>
          </div>
          <p role="status" className="min-h-5 text-sm">
            {notice}
          </p>
          {result && (
            <>
              <h3 className="font-medium">Start the server</h3>
              <pre tabIndex={0} aria-label="Server command">
                <code>{result.command}</code>
              </pre>
              <h3 className="font-medium">{result.websocketUrl ? "Check server health" : "Send a request"}</h3>
              <pre tabIndex={0} aria-label="Client command">
                <code>{result.clientCommand}</code>
              </pre>
              {result.websocketUrl && (
                <p className="break-all text-sm">
                  WebSocket URL: <code>{result.websocketUrl}</code>
                </p>
              )}
              <details open>
                <summary className="font-medium">Complete config.yaml</summary>
                <pre tabIndex={0} aria-label="Generated YAML">
                  <code>{result.yaml}</code>
                </pre>
              </details>
            </>
          )}
        </section>
      </div>
    </>
  );
}

export default function Cookbook() {
  const [index, setIndex] = useState<Index>();
  const [selected, setSelected] = useState("");
  const [catalog, setCatalog] = useState<Catalog>();
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);
  const id = useId();
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
        const deployments = data.models.flatMap((model) => model.deployments);
        setSelected(deployments.some((item) => item.id === requested) ? requested! : (deployments[0]?.id ?? ""));
      })
      .catch((error) => {
        if (!controller.signal.aborted) setError(errorText(error));
      });
    return () => controller.abort();
  }, [retry]);
  useEffect(() => {
    if (!index || !selected) return;
    const controller = new AbortController();
    setCatalog(undefined);
    setError("");
    api
      .loadDeployment(index, selected, { signal: controller.signal })
      .then(setCatalog)
      .catch((error) => {
        if (!controller.signal.aborted) setError(errorText(error));
      });
    return () => controller.abort();
  }, [index, selected]);
  function select(value: string) {
    setSelected(value);
    setCatalog(undefined);
    setError("");
    const url = new URL(window.location.href);
    url.searchParams.set("deployment", value);
    window.history.replaceState(null, "", url);
  }
  return (
    <div className="cookbook not-prose space-y-6">
      <div className="max-w-xl space-y-2">
        <label htmlFor={id} className="text-sm font-medium">
          Model and deployment
        </label>
        <select id={id} value={selected} disabled={!index} onChange={(event) => select(event.target.value)}>
          {!index && <option>Loading recipes…</option>}
          {index?.models.map((model) => (
            <optgroup key={model.id} label={model.title}>
              {model.deployments.map((deployment) => (
                <option key={deployment.id} value={deployment.id}>
                  {model.title} · {deployment.label}
                </option>
              ))}
            </optgroup>
          ))}
        </select>
      </div>
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
        <Editor key={catalog.id} catalog={catalog} />
      ) : (
        <p role="status">Loading deployment settings…</p>
      )}
    </div>
  );
}
