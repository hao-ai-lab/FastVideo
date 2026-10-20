import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, readdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { parse } from "yaml";
import { configAndLaunch, inferenceOutput, pythonSource, sdkCommand } from "../lib/cookbook-commands.ts";

test("every generated Python example remains valid after its source-backed settings are applied", () => {
  const directory = new URL("../.generated/source-examples/", import.meta.url);
  const examples = readdirSync(directory)
    .filter((name) => name.endsWith(".json"))
    .map((name) => {
      const data = JSON.parse(readFileSync(new URL(name, directory), "utf8"));
      const edits = Object.fromEntries(
        data.controls.map((field, i) => [
          i,
          typeof field.value === "number"
            ? 2
            : typeof field.value === "boolean"
              ? false
              : "A bowl 🥣 with 'quotes' and $(literal text)",
        ]),
      );
      return {
        name,
        baseline: pythonSource(data.source, data.controls),
        edited: pythonSource(data.source, data.controls, edits),
      };
    });
  assert.ok(examples.length > 25);
  execFileSync(
    "python3",
    [
      "-c",
      "import ast,json,sys\nfor example in json.load(sys.stdin):\n for key in ('baseline', 'edited'):\n  ast.parse(example[key], filename=example['name'])",
    ],
    { input: JSON.stringify(examples) },
  );
});

test("SDK launch commands have valid shell syntax without executing model code", () => {
  execFileSync("bash", ["-n"], {
    input: configAndLaunch("generator: {}\nrequest: {}\n", `FASTVIDEO_FA4=0 ${sdkCommand}`),
  });
});

test("copy-and-launch writes literal YAML before invoking the command", () => {
  const root = mkdtempSync(join(tmpdir(), "fastvideo-copy-command-"));
  try {
    const yaml = 'prompt: "$(touch injected) `touch injected2`"\nFASTVIDEO_CONFIG\n';
    execFileSync("bash", ["-c", configAndLaunch(yaml, "cp config.yaml received.yaml")], { cwd: root });
    assert.equal(readFileSync(join(root, "received.yaml"), "utf8"), yaml);
    assert.throws(() => readFileSync(join(root, "injected")));
    assert.throws(() => readFileSync(join(root, "injected2")));
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test("inference output preserves edited settings and hidden generator fields", () => {
  const config = {
    generator: {
      model_path: "FastVideo/Trim",
      engine: { num_gpus: 2 },
      pipeline: { experimental: { VSA_sparsity: 0.8 } },
    },
    server: { port: 9000 },
    default_request: { prompt: "A ceramic bowl", sampling: { num_frames: 243, seed: 0 } },
  };
  const result = inferenceOutput({ config, command: "FASTVIDEO_FA4=0 fastvideo serve --config config.yaml" });
  const saved = parse(result.yaml);
  assert.deepEqual(saved.generator, config.generator);
  assert.deepEqual(saved.request.sampling, config.default_request.sampling);
  assert.equal(saved.request.prompt, "A ceramic bowl");
  assert.equal(saved.request.output.save_video, true);
  assert.equal(saved.server, undefined);
  assert.equal(result.command, "FASTVIDEO_FA4=0 fastvideo generate --config config.yaml");
  assert.ok(config.server);
});

test("Wan I2V inference requires a source image and preserves the serving baseline", () => {
  const config = parse(
    readFileSync(new URL("../../examples/serving/openai_wan21_i2v_14b.yaml", import.meta.url), "utf8"),
  );
  const baseline = structuredClone(config);
  const result = { config, command: "fastvideo serve --config config.yaml" };
  assert.throws(() => inferenceOutput(result), /source image path/);
  assert.throws(() => inferenceOutput(result, "   "), /source image path/);
  const saved = parse(inferenceOutput(result, " /data/first-frame.png ").yaml);
  assert.equal(saved.request.inputs.image_path, "/data/first-frame.png");
  assert.deepEqual(saved.request.sampling, config.default_request.sampling);
  assert.deepEqual(config, baseline);
  assert.equal(config.default_request.inputs, undefined);
  const withDefaults = structuredClone(config);
  withDefaults.default_request.inputs = { image_path: "/data/existing.png", video_path: "/data/reference.mp4" };
  const existing = parse(inferenceOutput({ ...result, config: withDefaults }).yaml);
  assert.deepEqual(existing.request.inputs, withDefaults.default_request.inputs);
});
