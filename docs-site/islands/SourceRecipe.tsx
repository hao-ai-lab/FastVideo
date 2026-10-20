import { useEffect, useState } from "react";
import { Tabs } from "radix-ui";
import { parse, stringify } from "yaml";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import ModelHeader from "@/components/cookbook/ModelHeader";
import HardwarePicker from "@/components/cookbook/HardwarePicker";
import RecipeFacts from "@/components/cookbook/RecipeFacts";
import { configAndLaunch, fileAndLaunch, sdkCommand, pythonSource } from "@/lib/cookbook-commands";
import { sitePath, type DirectoryModel } from "@/lib/cookbook-pages";
import { api } from "@/lib/cookbook";
import {
  gpuCountFrom,
  readHardwareSelection,
  resolveHardware,
  useHardwareCatalog,
  writeHardwareSelection,
  type HardwareSelection,
} from "@/lib/hardware";

interface Control {
  name: string;
  label: string;
  value: string | number | boolean;
  spans?: number[][];
  path?: string[];
  flag?: string;
  integer?: boolean;
  booleanOptional?: boolean;
  choices?: (string | number | { value: string | number; label: string })[];
}
interface Example {
  source: string;
  controls: Control[];
}
const labels: Record<string, string> = {
  prompt: "Prompt",
  num_gpus: "GPU count",
  height: "Height",
  width: "Width",
  num_frames: "Frames",
  seed: "Seed",
  num_inference_steps: "Inference steps",
  guidance_scale: "Guidance scale",
  fps: "FPS",
  host: "Host",
  port: "Port",
  served_model_name: "Served model name",
};

/** Every source page uses the model editor; settings modify the maintained example itself. */
export default function SourceRecipe({ model }: { model: DirectoryModel }) {
  const [selected, setSelected] = useState(model.recipes[0].id);
  const [mode, setMode] = useState("sdk");
  const [loaded, setLoaded] = useState<{ url: string; example: Example }>();
  const [notice, setNotice] = useState("");
  const [showEnv, setShowEnv] = useState(false);
  const [showClient, setShowClient] = useState(false);
  const [outputTab, setOutputTab] = useState("commands");
  const [error, setError] = useState("");
  const [edits, setEdits] = useState<Record<number, string | number | boolean>>({});
  const [hardware, setHardware] = useState<HardwareSelection>(() =>
    readHardwareSelection(new URL(window.location.href).searchParams),
  );
  const hardwareCatalog = useHardwareCatalog();
  const recipe = model.recipes.find((recipe) => recipe.id === selected)!;
  const yamlSource =
    mode === "serving" ? recipe.serving?.source : /\.ya?ml$/.test(recipe.source) ? recipe.source : undefined;
  const sourceUrl = sitePath(
    yamlSource
      ? `/assets/cookbook-sources/${recipe.id}-${mode === "serving" ? "serving" : "inference"}.yaml`
      : `/assets/cookbook-examples/${recipe.id}.json`,
  );
  const example = loaded?.url === sourceUrl ? loaded.example : undefined;
  useEffect(() => {
    const controller = new AbortController();
    setError("");
    fetch(sourceUrl, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(`Example could not be loaded: HTTP ${response.status}.`);
        let data: Example;
        if (!yamlSource) data = await response.json();
        else {
          const source = await response.text();
          const config = parse(source);
          const controls: Control[] = [];
          function visit(value: Record<string, unknown>, path: string[] = []) {
            for (const [name, item] of Object.entries(value)) {
              if (item && typeof item === "object" && !Array.isArray(item))
                visit(item as Record<string, unknown>, [...path, name]);
              else if (labels[name] && ["string", "number", "boolean"].includes(typeof item))
                controls.push({
                  name,
                  label: labels[name],
                  value: item as Control["value"],
                  integer: !["guidance_scale", "audio_end_in_s"].includes(name),
                  path: [...path, name],
                });
            }
          }
          visit(config);
          data = { source, controls };
        }
        if (!yamlSource) {
          for (const knob of recipe.knobs ?? []) {
            const field = data.controls.find((field) => field.name === knob.key);
            if (field) field.choices = knob.options;
            else
              data.controls.push({
                name: knob.key,
                label: labels[knob.key] ?? knob.label,
                value: knob.default,
                flag: knob.flag,
                spans: [],
                choices: knob.options,
              });
          }
        }
        if (!controller.signal.aborted) setLoaded({ url: sourceUrl, example: data });
      })
      .catch((error) => {
        if (!controller.signal.aborted) setError(error.message);
      });
    return () => controller.abort();
  }, [sourceUrl, yamlSource, recipe]);
  function selectMode(next: string) {
    const reset = (next === "serving") !== (mode === "serving");
    setMode(next);
    if (reset) setEdits({});
    setNotice(reset ? "Serving uses a different native example, so generation edits were reset." : "");
  }
  async function copy(value: string, message = "Copied to clipboard.") {
    try {
      await navigator.clipboard.writeText(value);
      setNotice(message);
    } catch {
      setNotice("Copy failed. Select and copy the text below.");
    }
  }
  let source = example?.source ?? "";
  let failure = "";
  for (const [key, value] of Object.entries(edits)) {
    const field = example?.controls[Number(key)];
    if (
      field &&
      typeof field.value === "number" &&
      (!Number.isFinite(value) ||
        ((field.integer ?? Number.isInteger(field.value)) && !Number.isInteger(value)) ||
        (field.name === "num_gpus" && Number(value) < 1))
    )
      failure = "Enter valid numeric settings before copying the example.";
  }
  let config: Record<string, unknown> | undefined;
  if (example && yamlSource) {
    try {
      const root = parse(source) as Record<string, unknown>;
      config = root;
      for (const [key, value] of Object.entries(edits)) {
        const path = example.controls[Number(key)].path!;
        const parent = path
          .slice(0, -1)
          .reduce<Record<string, unknown>>((item, name) => item[name] as Record<string, unknown>, root);
        parent[path.at(-1)!] = value;
      }
      source = stringify(root);
    } catch {
      failure = "The checked-in YAML could not be parsed, so runnable output is withheld.";
    }
  } else if (example) {
    source = pythonSource(source, example.controls, edits);
  }
  const baselineCommand =
    mode === "serving" && recipe.serving
      ? `${recipe.serving.env ? `${recipe.serving.env} ` : ""}fastvideo serve --config ${recipe.serving.source}`
      : recipe.command;
  const localFile = recipe.source.replace(/[^/]+$/, `cookbook_${recipe.id.replaceAll("-", "_")}.py`);
  const command = yamlSource
    ? baselineCommand.replace(/--config\s+\S+/, "--config config.yaml")
    : baselineCommand.replace(recipe.source, localFile) +
      Object.entries(edits)
        .filter(([key]) => example?.controls[Number(key)].flag)
        .map(([key, value]) => {
          const field = example!.controls[Number(key)];
          return typeof value === "boolean" && field.booleanOptional
            ? ` ${value ? field.flag : field.flag!.replace(/^--/, "--no-")}`
            : ` ${field.flag} '${String(value).replaceAll("'", "'\"'\"'")}'`;
        })
        .join("");
  const sdk = mode === "sdk" && !!yamlSource;
  // The SDK consumes only generator + request; serving and CLI keep server and every native field.
  const displayedSource =
    config && sdk
      ? stringify({
          generator: config.generator as Record<string, unknown>,
          request: config.request ?? config.default_request,
        })
      : source;
  const launch = sdk ? command.replace(/fastvideo generate --config config\.yaml$/, sdkCommand) : command;
  const ready = !!example && !failure;
  const workload = recipe.task.toLowerCase().includes("image") ? "i2v" : "t2v";
  const clientCommand =
    mode === "serving" && ready
      ? api.sampleRequest(
          {
            runtime: { interface: "rest", server_defaults: { host: "127.0.0.1", port: 8000 } },
            model: { id: model.model },
            workload,
          },
          parse(source),
        ).command
      : undefined;
  const launchScript = yamlSource
    ? configAndLaunch(displayedSource, launch)
    : fileAndLaunch(localFile, displayedSource, launch);
  const environment = [...baselineCommand.matchAll(/(?:^|\s)([A-Z][A-Z0-9_]*=(?:'[^']*'|"[^"]*"|[^\s]+))/g)]
    .map((match) => match[1])
    .join("\n");
  const mlx = recipe.gpu_types.includes("Apple Silicon");
  const hardwareRecord = resolveHardware(hardware, hardwareCatalog.records);
  const gpuCount = config ? gpuCountFrom(config) : (recipe.hardware?.gpu_count ?? 1);
  const chooseHardware = (selection: HardwareSelection) => {
    setHardware(selection);
    const url = new URL(window.location.href);
    writeHardwareSelection(url, selection);
    window.history.replaceState(null, "", url);
  };
  const control = (field: Control, index: number) => {
    const value = Object.hasOwn(edits, index) ? edits[index] : field.value;
    return (
      <div className="matrix-field" key={index}>
        {field.choices || typeof value === "boolean" ? (
          <span className="matrix-field-label">{field.label}</span>
        ) : (
          <label htmlFor={`source-${index}`} className="matrix-field-label">
            {field.label}
          </label>
        )}
        {field.choices ? (
          <div role="group" aria-label={field.label} className="option-chips">
            {field.choices.map((option) => {
              const choice = typeof option === "object" ? option.value : option;
              return (
                <button
                  type="button"
                  key={String(choice)}
                  aria-pressed={value === choice}
                  onClick={() => setEdits({ ...edits, [index]: choice })}
                >
                  {typeof option === "object" ? option.label : String(option)}
                </button>
              );
            })}
          </div>
        ) : typeof value === "boolean" ? (
          <div role="group" aria-label={field.label} className="option-chips">
            {[true, false].map((choice) => (
              <button
                type="button"
                key={String(choice)}
                aria-pressed={value === choice}
                onClick={() => setEdits({ ...edits, [index]: choice })}
              >
                {choice ? "On" : "Off"}
              </button>
            ))}
          </div>
        ) : (
          <Input
            id={`source-${index}`}
            type={typeof field.value === "number" ? "number" : "text"}
            value={typeof value === "number" && Number.isNaN(value) ? "" : String(value)}
            step={typeof field.value === "number" && (field.integer ?? Number.isInteger(field.value)) ? 1 : "any"}
            onChange={(event) =>
              setEdits({
                ...edits,
                [index]:
                  typeof field.value === "number"
                    ? event.target.value === ""
                      ? NaN
                      : Number(event.target.value)
                    : event.target.value,
              })
            }
          />
        )}
      </div>
    );
  };
  return (
    <div className="cookbook not-prose recipe-detail">
      <ModelHeader
        model={model}
        actions={
          <>
            <Button
              variant="outline"
              disabled={!ready}
              onClick={() => copy(displayedSource, yamlSource ? "Copied config.yaml." : "Copied Python example.")}
            >
              Copy {yamlSource ? "YAML" : "Python"}
            </Button>
            <Button
              disabled={!ready}
              onClick={() => {
                const url = URL.createObjectURL(new Blob([displayedSource], { type: "text/plain" }));
                const link = document.createElement("a");
                link.href = url;
                link.download = yamlSource ? "config.yaml" : localFile.split("/").at(-1)!;
                link.click();
                setTimeout(() => URL.revokeObjectURL(url), 1000);
              }}
            >
              Download {yamlSource ? "YAML" : "Python"}
            </Button>
          </>
        }
      />
      <div className="recipe-run-modes option-chips" role="group" aria-label="Run mode">
        {["sdk", "inference", ...(recipe.serving ? ["serving"] : [])].map((entry) => (
          <button type="button" key={entry} aria-pressed={mode === entry} onClick={() => selectMode(entry)}>
            {entry === "sdk" ? "Python SDK" : entry === "inference" ? "CLI" : "Serving"}
          </button>
        ))}
      </div>
      <section className="recipe-command-box" aria-label="Source recipe">
        <Tabs.Root value={outputTab} onValueChange={setOutputTab}>
          <div className="recipe-command-toolbar">
            <div className="recipe-command-tools">
              <button
                type="button"
                disabled={!ready}
                onClick={() =>
                  copy(
                    launchScript,
                    yamlSource ? "Copied config.yaml and launch command." : "Copied example and launch command.",
                  )
                }
              >
                {yamlSource ? "Copy config + command" : "Copy example + command"}
              </button>
              {mode === "serving" && (
                <button
                  type="button"
                  disabled={!clientCommand}
                  aria-expanded={showClient}
                  aria-controls="source-client-command"
                  onClick={() => {
                    setOutputTab("commands");
                    setShowClient(!showClient);
                  }}
                >
                  Show request
                </button>
              )}
              {environment && (
                <button type="button" aria-pressed={showEnv} onClick={() => setShowEnv(!showEnv)}>
                  Show environment
                </button>
              )}
            </div>
            <Tabs.List className="recipe-tabs" aria-label="Output format">
              <Tabs.Trigger value="commands">Commands</Tabs.Trigger>
              <Tabs.Trigger value="source">{yamlSource ? "config.yaml" : "Python example"}</Tabs.Trigger>
            </Tabs.List>
          </div>
          {showEnv && (
            <pre className="recipe-environment" tabIndex={0} aria-label="Environment variables">
              <code>{environment}</code>
            </pre>
          )}
          {(error || failure) && (
            <p className="recipe-error" role="alert">
              {error || failure}
            </p>
          )}
          <Tabs.Content value="commands" className="recipe-command-content">
            <div className="recipe-code-heading">
              <span>{mode === "serving" ? "Serve" : mode === "sdk" ? "Python SDK" : "Generate"}</span>
            </div>
            <pre tabIndex={0} aria-label={mode === "serving" ? "Server command" : "Inference command"}>
              <code>{launch}</code>
            </pre>
            <p className="matrix-note">
              Uses your settings; the copy action includes the source file and launch command. Run from an installed
              FastVideo checkout.
            </p>
            {clientCommand && (
              <details
                className="recipe-request-code"
                id="source-client-command"
                open={showClient}
                onToggle={(event) => setShowClient(event.currentTarget.open)}
              >
                <summary>Example request</summary>
                <button type="button" onClick={() => copy(clientCommand, "Copied example request.")}>
                  Copy request command
                </button>
                <pre tabIndex={0} aria-label="Client command">
                  <code>{clientCommand}</code>
                </pre>
              </details>
            )}
          </Tabs.Content>
          <Tabs.Content value="source" className="recipe-command-content">
            <pre tabIndex={0} aria-label="Source example">
              <code>{displayedSource || "Loading example…"}</code>
            </pre>
          </Tabs.Content>
        </Tabs.Root>
        <p role="status" className="recipe-copy-status">
          {notice}
        </p>
      </section>
      <details className="recipe-install">
        <summary>
          Installation and setup <span>{mlx ? "MLX" : "CUDA"}</span>
        </summary>
        {mode === "serving" && recipe.serving && (
          <pre>
            <code>{recipe.serving.install}</code>
          </pre>
        )}
        {mode === "serving" && recipe.serving?.prepare && (
          <>
            <pre>
              <code>{recipe.serving.prepare}</code>
            </pre>
            <Button variant="outline" onClick={() => copy(recipe.serving!.prepare!)}>
              Copy preparation commands
            </Button>
          </>
        )}
        <a href={sitePath(model.familyPath)}>Model setup guide ↗</a>
      </details>
      <section className="recipe-matrix" aria-label="Configuration settings">
        <div className="matrix-row">
          <h2>Hardware</h2>
          <div className="matrix-row-content">
            <div className="hardware-backends option-chips" role="group" aria-label="Hardware backend">
              {[
                ...new Set(model.recipes.map((entry) => (entry.gpu_types.includes("Apple Silicon") ? "MLX" : "CUDA"))),
              ].map((platform) => (
                <button
                  type="button"
                  key={platform}
                  aria-pressed={platform === (mlx ? "MLX" : "CUDA")}
                  onClick={() => {
                    if (platform === (mlx ? "MLX" : "CUDA")) return;
                    const target = model.recipes.find((entry) =>
                      entry.gpu_types.includes(platform === "MLX" ? "Apple Silicon" : "NVIDIA"),
                    )!;
                    setSelected(target.id);
                    setMode("sdk");
                    setEdits({});
                    setNotice(`Switched to the ${platform} recipe; generation edits were reset.`);
                  }}
                >
                  {platform}
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
                {example?.controls.map((field, i) => (field.name === "num_gpus" ? control(field, i) : null))}
                {!example?.controls.some((field) => field.name === "num_gpus") && (
                  <p>{mlx ? "1 Apple Silicon host" : `${recipe.hardware?.gpu_count ?? 1} GPU in the source example`}</p>
                )}
                <a href={sitePath(mlx ? "/getting_started/installation/mlx" : "/getting_started/installation/gpu")}>
                  Setup guide ↗
                </a>
              </div>
            </div>
            <RecipeFacts
              validity={failure ? "invalid" : example ? "valid" : "loading"}
              validityText={yamlSource ? "YAML parses" : "Example settings applied"}
              validityDetail="This page checks that the YAML parses and that edited numeric fields are finite and supported by the example. It does not run the full recipe schema or runtime topology validation."
              fit="unknown"
              evidence="none"
              subject={yamlSource ? "configuration" : "example"}
            />
          </div>
        </div>
        {model.recipes.filter((entry) => entry.gpu_types.includes(mlx ? "Apple Silicon" : "NVIDIA")).length > 1 && (
          <div className="matrix-row">
            <h2>Recipe</h2>
            <div className="matrix-row-content option-chips" role="group" aria-label="Recipe variant">
              {model.recipes
                .filter((entry) => entry.gpu_types.includes(mlx ? "Apple Silicon" : "NVIDIA"))
                .map((entry) => (
                  <button
                    type="button"
                    key={entry.id}
                    aria-pressed={selected === entry.id}
                    onClick={() => {
                      setNotice("");
                      setSelected(entry.id);
                      setMode("sdk");
                      setEdits({});
                    }}
                  >
                    {entry.label}
                  </button>
                ))}
            </div>
          </div>
        )}
        <div className="matrix-row">
          <h2>Generation settings</h2>
          <div className="matrix-row-content">
            <div className="matrix-options-grid">
              {example?.controls.map((field, i) =>
                field.name !== "num_gpus" && field.path?.[0] !== "server" ? control(field, i) : null,
              )}
            </div>
            {example && !example.controls.length && (
              <p className="matrix-note">
                This example uses its model defaults. Open the Python example to customize its pipeline.
              </p>
            )}
          </div>
        </div>
        {example?.controls.some((field) => field.path?.[0] === "server") && (
          <details className="matrix-advanced">
            <summary>
              Advanced <span>Server settings</span>
            </summary>
            <div className="matrix-row">
              <h2>Server</h2>
              <div className="matrix-row-content matrix-options-grid">
                {example.controls.map((field, i) => (field.path?.[0] === "server" ? control(field, i) : null))}
              </div>
            </div>
          </details>
        )}
      </section>
      <div className="recipe-bottom-actions">
        <a
          href={`https://github.com/hao-ai-lab/FastVideo/blob/main/${mode === "serving" ? recipe.serving?.source : recipe.source}`}
        >
          View original example ↗
        </a>
        <Button
          variant="ghost"
          onClick={() => {
            setEdits({});
            setNotice("Restored the baseline example.");
          }}
        >
          Reset changes
        </Button>
      </div>
    </div>
  );
}
