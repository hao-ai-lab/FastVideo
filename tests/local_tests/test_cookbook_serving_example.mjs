import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { spawnSync } from "node:child_process";
import test from "node:test";

const require = createRequire(import.meta.url);
const { resolveRecipe, shellQuote, configArguments } = require("../../docs/assets/cookbook-serving.js");
const realBundle = () => JSON.parse(readFileSync(new URL("../../docs/assets/cookbook-serving-example.json", import.meta.url), "utf8"));

function fixture() {
  return {
    defaults: {
      runtimes: { cuda: { command: ["fastvideo", "serve"] } },
      tasks: { t2v: { client: "video", options: ["port", "num_gpus", "offload"] } },
      options: {
        port: { label: "Port", path: "server.port", type: "integer", min: 1, max: 65535, default: 8000 },
        num_gpus: { label: "GPU count", path: "generator.engine.num_gpus", type: "integer", choices: [1, 2, 4, 8], default: 1 },
        offload: { path: "generator.engine.offload.vae", type: "boolean", default: true },
      },
    },
    hardware: { gpu: { label: "Synthetic GPU type", runtime: "cuda" } },
    recipes: [{
      id: "example", model_id: "Example/Model", runtime: "cuda", task: "t2v",
      default_hardware: "gpu", hardware: ["gpu"], install: "uv pip install -e .",
      settings: { server: { host: "0.0.0.0", served_model_name: "example", output_dir: "" } },
      example_request: { prompt: "A fox in snow", num_inference_steps: 9, seed: 42 },
      options: { num_gpus: { choices: [4], default: 4, also_set: ["generator.engine.parallelism.sp_size"] } },
    }],
  };
}

test("model differences replace choices while inheriting paths, types, ranges and labels", () => {
  const input = fixture();
  input.recipes[0].options.port = { default: 9000 };
  const before = JSON.stringify(input);
  const result = resolveRecipe(input, "example");
  const port = result.options.find((option) => option.key === "port");
  assert.deepEqual(result.options.find((option) => option.key === "num_gpus").choices, [4]);
  assert.equal(port.label, "Port");
  assert.equal(port.path, "server.port");
  assert.equal(port.type, "integer");
  assert.equal(port.min, 1);
  assert.equal(port.max, 65535);
  assert.equal(port.default, 9000);
  assert.equal(result.config.generator.engine.num_gpus, 4);
  assert.equal(result.config.generator.engine.parallelism.sp_size, 4);
  assert.equal(JSON.stringify(input), before, "resolution must not mutate metadata or static requests");
});

test("value precedence and origin are shared, model, then explicit user", () => {
  const input = fixture();
  input.recipes[0].options.port = { default: 9000 };
  const result = resolveRecipe(input, "example", { values: { port: 9010 } });
  const options = Object.fromEntries(result.options.map((option) => [option.key, option]));
  assert.equal(options.offload.source, "shared");
  assert.equal(options.num_gpus.source, "model");
  assert.equal(options.port.default, 9000);
  assert.equal(options.port.defaultSource, "model");
  assert.equal(options.port.value, 9010);
  assert.equal(options.port.source, "user");
});

test("GPU count changes independently of the selected GPU type and updates linked SP", () => {
  const input = fixture();
  input.recipes[0].options.num_gpus.choices = [2, 4];
  for (const count of [2, 4]) {
    const result = resolveRecipe(input, "example", { hardware: "gpu", values: { num_gpus: count } });
    assert.equal(result.hardware.label, "Synthetic GPU type");
    assert.equal(result.config.generator.engine.num_gpus, count);
    assert.equal(result.config.generator.engine.parallelism.sp_size, count);
  }
});

test("invalid selections fail instead of generating a runnable-looking command", () => {
  for (const values of [{ port: 0 }, { port: 65536 }, { port: 8.5 }, { port: "8000" },
    { port: NaN }, { num_gpus: 2 }, { offload: "false" }, { mystery: 1 }]) {
    assert.throws(() => resolveRecipe(fixture(), "example", { values }));
  }
  assert.throws(() => resolveRecipe(fixture(), "missing"), /Unknown recipe/);
  assert.throws(() => resolveRecipe(fixture(), "example", { hardware: "other" }), /Unsupported hardware/);
  assert.throws(() => resolveRecipe(fixture(), "example", { port: 9000 }), /Selections/);
  assert.throws(() => resolveRecipe(fixture(), "example", { request: { prompt: "Changed" } }), /Selections/);
  const wrongRuntime = fixture();
  wrongRuntime.hardware.gpu.runtime = "mlx";
  assert.throws(() => resolveRecipe(wrongRuntime, "example"), /Unsupported hardware/);
});

test("invalid recipe defaults cannot be hidden by a valid user override", () => {
  const input = fixture();
  input.recipes[0].options.num_gpus.default = 2;
  assert.throws(() => resolveRecipe(input, "example", { values: { num_gpus: 4 } }), /num_gpus must be one of/);
});

test("model deployment controls support numeric ranges and unsupported controls reject selections", () => {
  const input = fixture();
  input.recipes[0].options.offload = { supported: false };
  input.recipes[0].options.sparsity = {
    path: "generator.pipeline.experimental.VSA_sparsity", type: "number", min: 0, max: 1, default: 0.8,
  };
  const result = resolveRecipe(input, "example", { values: { sparsity: 0.6 } });
  assert.equal(result.options.some((option) => option.key === "offload"), false);
  assert.equal(result.config.generator.pipeline.experimental.VSA_sparsity, 0.6);
  assert.throws(() => resolveRecipe(input, "example", { values: { sparsity: 1.1 } }), /at most 1/);
  assert.throws(() => resolveRecipe(input, "example", { values: { offload: true } }), /unsupported option/);
});

test("deployment metadata cannot introduce request options or default_request settings", () => {
  const input = fixture();
  input.recipes[0].options.steps = { path: "default_request.sampling.num_inference_steps", type: "integer", default: 9 };
  assert.throws(() => resolveRecipe(input, "example"), /Invalid serving configuration path/);
  delete input.recipes[0].options.steps;
  input.recipes[0].settings.default_request = { prompt: "hidden request default" };
  assert.throws(() => resolveRecipe(input, "example"), /only generator and server/);
});

test("port and model alias stay synchronized without changing static request values", () => {
  const input = fixture();
  input.recipes[0].settings.server.served_model_name = "model alias";
  input.recipes[0].example_request.model = "old alias";
  const result = resolveRecipe(input, "example", { values: { port: 9000 } });
  assert.match(result.command, /--server.port 9000/);
  assert.match(result.command, /--server.served_model_name 'model alias'/);
  assert.equal(result.healthCommand, "curl --fail-with-body http://127.0.0.1:9000/health");
  assert.equal(result.clientRequest.endpoint, "http://127.0.0.1:9000/v1/videos/sync");
  assert.deepEqual(result.clientRequest.body, { model: "model alias", prompt: "A fox in snow", num_inference_steps: 9, seed: 42 });
  assert.equal(result.clientRequest.outputFile, "output.mp4");
  assert.equal(input.recipes[0].example_request.model, "old alias");
  assert.equal(result.config.server.host, "0.0.0.0");
  assert.equal(result.command.includes("num_inference_steps"), false);
});

test("static requests require a prompt and I2V requires a server-local image path or URL", () => {
  for (const prompt of [undefined, null, "", "  ", 3]) {
    const input = fixture();
    input.recipes[0].example_request.prompt = prompt;
    assert.throws(() => resolveRecipe(input, "example"), /example_request.prompt/);
  }
  const input = fixture();
  input.defaults.tasks.t2v.requires_image = true;
  for (const reference of [undefined, "", "  ", null, 7]) {
    input.recipes[0].example_request.input_reference = reference;
    assert.throws(() => resolveRecipe(input, "example"), /example_request.input_reference/);
  }
  input.recipes[0].example_request.input_reference = "/path/to/first-frame.png";
  const result = resolveRecipe(input, "example");
  assert.equal(result.clientRequest.body.input_reference, "/path/to/first-frame.png");
  assert.equal(result.command.includes("input_reference"), false);
});

test("image test request retains static sampling values and requests PNG before decoding", () => {
  const input = fixture();
  input.defaults.tasks.t2v.client = "image";
  Object.assign(input.recipes[0].example_request, { size: "768x512", guidance_scale: 0, negative_prompt: "blur" });
  const result = resolveRecipe(input, "example");
  assert.deepEqual(result.clientRequest.body, {
    model: "example", prompt: "A fox in snow", response_format: "b64_json", output_format: "png", size: "768x512",
    seed: 42, num_inference_steps: 9, guidance_scale: 0, negative_prompt: "blur",
  });
  assert.match(result.clientRequest.endpoint, /\/v1\/images\/generations$/);
  assert.equal(result.clientRequest.outputFile, "output.png");
  assert.match(result.clientCommand, /--output response.json &&/);
  assert.match(result.clientCommand, /python -c/);
  assert.match(result.clientCommand, /base64.b64decode/);
});

test("empty or URL-shaped bind hosts cannot generate an invalid client address", () => {
  for (const host of ["", " ", "localhost/path", "http://localhost", "a\nb", "true", "1234", "[::1]"]) {
    const input = fixture();
    input.recipes[0].settings.server.host = host;
    assert.throws(() => resolveRecipe(input, "example"), /hostname or IP address/);
  }
  const ipv6 = fixture();
  ipv6.recipes[0].settings.server.host = "::1";
  assert.match(resolveRecipe(ipv6, "example").healthCommand, /http:\/\/\[::1\]:8000\/health/);
});

test("shell quoting preserves empty strings, quotes and executable-looking text as literal arguments", () => {
  const values = ["", "it's a test", "$(touch /tmp/cookbook-should-not-execute)", "a\nb", "`date`", "--value", "[1,true]"];
  // The probe only prints arguments; shellQuote must prevent substitutions in those arguments.
  const probe = spawnSync("/bin/sh", ["-c", `printf '%s\\0' ${values.map(shellQuote).join(" ")}`], { encoding: "utf8" });
  assert.equal(probe.status, 0);
  assert.deepEqual(probe.stdout.split("\0").slice(0, -1), values);
});

test("serialization includes every deployment leaf including false, empty strings, arrays and empty objects", () => {
  assert.deepEqual(configArguments({ generator: { enabled: false, text: "", list: [1, true], map: {} } }), [
    "--generator.enabled", "false", "--generator.text", "", "--generator.list", "[1,true]", "--generator.map", "{}",
  ]);
  const input = fixture();
  input.recipes[0].settings.generator = { pipeline: { experimental: { flags: [1, true] } } };
  const result = resolveRecipe(input, "example");
  assert.match(result.command, /--server.output_dir ''/);
  assert.match(result.command, /--generator.pipeline.experimental.flags '\[1,true\]'/);
  assert.equal(result.command.includes("--config"), false);
  assert.deepEqual(result.argv.slice(0, 2), ["fastvideo", "serve"]);
});

test("three real recipes expose deployment-only controls and keep static requests separate", () => {
  const input = realBundle();
  const original = JSON.stringify(input);
  resolveRecipe(input, "fasth3-v2", { values: { port: 9000, vae_offload: false } });
  for (const [id, count, steps] of [["fasth3-v2", 4, 9], ["zimage-turbo", 1, 8], ["wan21-i2v", 2, 40]]) {
    const result = resolveRecipe(input, id);
    assert.equal(result.config.generator.engine.num_gpus, count);
    assert.equal(result.config.generator.engine.parallelism.sp_size, count);
    assert.equal(result.config.server.port, 8000);
    assert.equal(Object.hasOwn(result.config, "default_request"), false);
    assert.equal(result.command.includes("--default_request."), false);
    assert.equal(result.command.includes("examples/serving/"), false);
    assert.equal(result.clientRequest.body.num_inference_steps, steps);
    assert.deepEqual(result.options.map((option) => option.key).sort(),
      ["host", "port", "num_gpus", "vae_offload", ...(id === "fasth3-v2" ? ["vsa_sparsity"] : [])].sort());
    for (const key of ["seed", "num_inference_steps", "guidance_scale", "num_frames", "prompt"]) {
      assert.throws(() => resolveRecipe(input, id, { values: { [key]: 1 } }), /Unknown or unsupported option/);
    }
    assert.equal(Object.hasOwn(result.clientRequest.body, "input_reference"), id === "wan21-i2v");
  }
  const h3 = resolveRecipe(input, "fasth3-v2");
  assert.equal(h3.clientRequest.body.num_frames, 124);
  assert.equal(h3.clientRequest.body.seed, 1000);
  assert.equal(h3.config.generator.pipeline.experimental.VSA_sparsity, 0.8);
  const image = resolveRecipe(input, "zimage-turbo");
  assert.equal(image.clientRequest.body.guidance_scale, 0);
  assert.equal(image.clientRequest.body.size, "1024x1024");
  assert.equal(image.clientRequest.body.output_format, "png");
  const wan = resolveRecipe(input, "wan21-i2v");
  assert.equal(wan.clientRequest.body.num_frames, 77);
  assert.equal(wan.clientRequest.body.input_reference, "/path/to/first-frame.png");
  assert.equal(Object.hasOwn(wan.clientRequest.body, "task"), false);
  assert.equal(JSON.stringify(input), original);
});

test("real model partial overrides preserve shared properties and deployment coupling", () => {
  const input = realBundle();
  const h3 = resolveRecipe(input, "fasth3-v2", { values: { num_gpus: 2, vsa_sparsity: 0.5 } });
  const count = h3.options.find((option) => option.key === "num_gpus");
  for (const property of ["path", "type", "label", "choices"]) {
    assert.deepEqual(count[property], input.defaults.options.num_gpus[property]);
  }
  assert.equal(count.default, 4);
  assert.equal(h3.config.generator.engine.num_gpus, 2);
  assert.equal(h3.config.generator.engine.parallelism.sp_size, 2);
  assert.equal(h3.config.generator.pipeline.experimental.VSA_sparsity, 0.5);
  const wan = resolveRecipe(input, "wan21-i2v", { values: { num_gpus: 4 } });
  assert.deepEqual(wan.options.find((option) => option.key === "num_gpus").choices, [2, 4, 8]);
  assert.equal(wan.config.generator.engine.parallelism.sp_size, 4);
  assert.equal(wan.config.generator.engine.parallelism.tp_size, 2);
  assert.throws(() => resolveRecipe(input, "zimage-turbo", { values: { num_gpus: 2 } }), /must be one of/);
  const edited = resolveRecipe(input, "fasth3-v2", { values: { port: 9000, vae_offload: false } });
  assert.match(edited.command, /--server.port 9000/);
  assert.match(edited.command, /--generator.engine.offload.vae false/);
  assert.match(edited.healthCommand, /:9000\/health/);
  assert.match(edited.clientCommand, /:9000\/v1\/videos\/sync/);
  assert.deepEqual(edited.clientRequest.body, resolveRecipe(input, "fasth3-v2").clientRequest.body);
});
