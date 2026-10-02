import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { spawnSync } from "node:child_process";
import test from "node:test";

const require = createRequire(import.meta.url);
const { resolveRecipe, shellQuote, configArguments } = require("../../docs/assets/cookbook-serving.js");

function fixture() {
  return {
    defaults: {
      runtimes: { cuda: { command: ["fastvideo", "serve"] } },
      tasks: { t2v: { client: "video", options: ["port", "num_gpus", "steps", "seed", "offload"] } },
      options: {
        port: { label: "Port", path: "server.port", type: "integer", min: 1, max: 65535, default: 8000 },
        num_gpus: { path: "generator.engine.num_gpus", type: "integer", choices: [1, 2, 4, 8], default: 1 },
        steps: { path: "default_request.sampling.num_inference_steps", type: "integer", min: 1, default: 50 },
        seed: { path: "default_request.sampling.seed", type: "integer", min: 0, default: 42 },
        offload: { path: "generator.engine.offload.vae", type: "boolean", default: true },
      },
    },
    hardware: { gpu: { label: "Synthetic GPU type", runtime: "cuda" } },
    recipes: [{
      id: "example", model_id: "Example/Model", runtime: "cuda", task: "t2v",
      default_hardware: "gpu", hardware: ["gpu"], install: "uv pip install -e .",
      settings: { server: { host: "0.0.0.0", served_model_name: "example" }, default_request: { negative_prompt: "" } },
      options: {
        num_gpus: { choices: [4], default: 4, also_set: ["generator.engine.parallelism.sp_size"] },
        steps: { choices: [9], default: 9 },
        seed: { default: 1000 },
      },
    }],
  };
}

test("model differences replace choices while inheriting paths, types, ranges and labels", () => {
  const input = fixture();
  const before = JSON.stringify(input);
  const result = resolveRecipe(input, "example");
  assert.deepEqual(result.options.find((option) => option.key === "num_gpus").choices, [4]);
  assert.equal(result.options.find((option) => option.key === "steps").min, 1);
  assert.equal(result.options.find((option) => option.key === "port").label, "Port");
  assert.equal(result.config.generator.engine.num_gpus, 4);
  assert.equal(result.config.generator.engine.parallelism.sp_size, 4);
  assert.equal(result.config.default_request.sampling.num_inference_steps, 9);
  assert.equal(JSON.stringify(input), before, "resolution must not mutate the source metadata");
});

test("value precedence and origin are shared, model, then explicit user", () => {
  const result = resolveRecipe(fixture(), "example", { values: { seed: 2, offload: false } });
  const options = Object.fromEntries(result.options.map((option) => [option.key, option]));
  assert.equal(options.port.source, "shared");
  assert.equal(options.num_gpus.source, "model");
  assert.equal(options.steps.source, "model");
  assert.equal(options.seed.default, 1000);
  assert.equal(options.seed.defaultSource, "model");
  assert.equal(options.seed.value, 2);
  assert.equal(options.seed.source, "user");
  assert.equal(result.config.generator.engine.offload.vae, false);
});

test("a synthetic model can select two counts independently of its single GPU type", () => {
  // Resolver behavior only: this does not claim two-GPU support for FastH3 V2.
  const input = fixture();
  input.recipes[0].options.num_gpus.choices = [2, 4];
  for (const count of [2, 4]) {
    const result = resolveRecipe(input, "example", { hardware: "gpu", values: { num_gpus: count } });
    assert.equal(result.hardware.label, "Synthetic GPU type");
    assert.equal(result.config.generator.engine.num_gpus, count);
    assert.equal(result.config.generator.engine.parallelism.sp_size, count);
    assert.match(result.command, new RegExp(`--generator.engine.num_gpus ${count}`));
    assert.match(result.command, new RegExp(`--generator.engine.parallelism.sp_size ${count}`));
  }
});

test("invalid selections fail instead of generating a runnable-looking command", () => {
  for (const values of [{ port: 0 }, { port: 65536 }, { port: 8.5 }, { port: "8000" },
    { port: NaN }, { steps: 8 }, { num_gpus: 2 }, { offload: "false" }, { mystery: 1 }]) {
    assert.throws(() => resolveRecipe(fixture(), "example", { values }));
  }
  assert.throws(() => resolveRecipe(fixture(), "missing"), /Unknown recipe/);
  assert.throws(() => resolveRecipe(fixture(), "example", { hardware: "other" }), /Unsupported hardware/);
  assert.throws(() => resolveRecipe(fixture(), "example", { port: 9000 }), /Selections/);
  const wrongRuntime = fixture();
  wrongRuntime.hardware.gpu.runtime = "mlx";
  assert.throws(() => resolveRecipe(wrongRuntime, "example"), /Unsupported hardware/);
});

test("invalid recipe defaults cannot be hidden by a valid user override", () => {
  const input = fixture();
  input.recipes[0].options.steps.default = 8;
  assert.throws(() => resolveRecipe(input, "example", { values: { steps: 9 } }), /steps must be one of/);
});

test("model-specific controls use the same mapping and unsupported controls reject selections", () => {
  const input = fixture();
  input.recipes[0].options.offload = { supported: false };
  input.recipes[0].options.sparsity = {
    path: "generator.pipeline.experimental.VSA_sparsity", type: "number", min: 0, max: 1, default: 0.8,
  };
  const result = resolveRecipe(input, "example", { values: { sparsity: 0.6 } });
  assert.equal(result.options.some((option) => option.key === "offload"), false);
  assert.equal(result.config.generator.pipeline.experimental.VSA_sparsity, 0.6);
  assert.throws(() => resolveRecipe(input, "example", { values: { offload: true } }), /unsupported option/);
});

test("port and model alias stay synchronized, with a connectable wildcard address", () => {
  const input = fixture();
  input.recipes[0].settings.server.served_model_name = "model alias";
  const result = resolveRecipe(input, "example", { values: { port: 9000 } });
  assert.match(result.command, /--server.port 9000/);
  assert.match(result.command, /--server.served_model_name 'model alias'/);
  assert.equal(result.healthCommand, "curl --fail-with-body http://127.0.0.1:9000/health");
  assert.match(result.clientCommand, /http:\/\/127\.0\.0\.1:9000\/v1\/videos\/sync/);
  assert.match(result.clientCommand, /"model":"model alias"/);
  assert.match(result.clientCommand, /--output output.mp4/);
  assert.equal(result.config.server.host, "0.0.0.0", "client normalization must not alter the server bind address");
  assert.equal(result.clientRequest.endpoint, "http://127.0.0.1:9000/v1/videos/sync");
  assert.equal(result.clientRequest.body.model, "model alias");
  assert.equal(result.clientRequest.outputFile, "output.mp4");
});

test("prompt is request-only and invalid or unknown request fields are rejected", () => {
  const result = resolveRecipe(fixture(), "example", { request: { prompt: "A fox's jump" } });
  assert.equal(result.clientRequest.body.prompt, "A fox's jump");
  assert.equal(result.command.includes("jump"), false);
  for (const request of [null, [], { prompt: "" }, { prompt: "  " }, { prompt: 3 },
    { prompt: null }, { input_reference: "/tmp/frame.png" }, { extra: true }]) {
    assert.throws(() => resolveRecipe(fixture(), "example", { request }));
  }
});

test("image-to-video requires a reference and sends it only in the client request", () => {
  const input = fixture();
  Object.assign(input.defaults.tasks.t2v, {
    requires_image: true, input_reference: "/path/to/first-frame.png", prompt: "Animate the image",
  });
  assert.equal(resolveRecipe(input, "example").clientRequest.body.input_reference, "/path/to/first-frame.png");
  const result = resolveRecipe(input, "example", { request: { input_reference: "https://example.com/frame.png" } });
  assert.deepEqual(result.clientRequest.body, {
    model: "example", prompt: "Animate the image", input_reference: "https://example.com/frame.png",
  });
  assert.equal(result.command.includes("input_reference"), false);
  for (const reference of ["", "  ", null, 7]) {
    assert.throws(() => resolveRecipe(input, "example", { request: { input_reference: reference } }), /input_reference/);
  }
  delete input.defaults.tasks.t2v.input_reference;
  assert.throws(() => resolveRecipe(input, "example"), /input_reference/);
});

test("image client explicitly carries resolved sampling and requests PNG before decoding", () => {
  const input = fixture();
  input.defaults.tasks.t2v.client = "image";
  Object.assign(input.recipes[0].settings.default_request, {
    sampling: { width: 768, height: 512, guidance_scale: 0 }, negative_prompt: "blur",
  });
  const result = resolveRecipe(input, "example", { values: { seed: 19 }, request: { prompt: "A lantern" } });
  assert.deepEqual(result.clientRequest.body, {
    model: "example", prompt: "A lantern", response_format: "b64_json", output_format: "png", size: "768x512",
    seed: 19, num_inference_steps: 9, guidance_scale: 0, negative_prompt: "blur",
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

test("shell quoting round-trips empty strings, quotes and executable-looking text as literal arguments", () => {
  const values = ["", "it's a test", "$(touch /tmp/cookbook-should-not-execute)", "a\nb", "`date`", "--value", "[1,true]"];
  // The probe only prints arguments; shellQuote must prevent substitutions in those arguments.
  const probe = spawnSync("/bin/sh", ["-c", `printf '%s\\0' ${values.map(shellQuote).join(" ")}`], { encoding: "utf8" });
  assert.equal(probe.status, 0);
  assert.deepEqual(probe.stdout.split("\0").slice(0, -1), values);
});

test("serialization includes every configured leaf including false, empty strings, arrays and empty objects", () => {
  assert.deepEqual(configArguments({ generator: { enabled: false, text: "", list: [1, true], map: {} } }), [
    "--generator.enabled", "false", "--generator.text", "", "--generator.list", "[1,true]", "--generator.map", "{}",
  ]);
  const input = fixture();
  input.recipes[0].settings.generator = { pipeline: { experimental: { flags: [1, true] } } };
  const result = resolveRecipe(input, "example");
  assert.match(result.command, /--default_request.negative_prompt ''/);
  assert.match(result.command, /--generator.pipeline.experimental.flags '\[1,true\]'/);
  assert.equal(result.command.includes("--config"), false);
  assert.deepEqual(result.argv.slice(0, 2), ["fastvideo", "serve"]);
});

test("generated real-model metadata produces an explicit FastH3 V2 command", () => {
  const url = new URL("../../docs/assets/cookbook-serving-example.json", import.meta.url);
  const input = JSON.parse(readFileSync(url, "utf8"));
  const result = resolveRecipe(input, "fasth3-v2");
  assert.equal(result.recipe.default_hardware, "gb200");
  assert.equal(result.hardware.label, "NVIDIA GB200");
  assert.equal(Object.hasOwn(result.hardware, "gpu_count"), false);
  assert.equal(Object.hasOwn(result.hardware, "defaults"), false);
  assert.deepEqual(result.options.find((option) => option.key === "num_gpus").choices, [4]);
  assert.throws(() => resolveRecipe(input, "fasth3-v2", { values: { num_gpus: 2 } }), /num_gpus must be one of/);
  assert.equal(result.config.generator.model_path, "FastVideo/FastVideo-FastH3-8-Step-V2");
  assert.equal(result.config.generator.engine.num_gpus, 4);
  assert.equal(result.config.generator.engine.parallelism.sp_size, 4);
  assert.equal(result.config.generator.engine.parallelism.tp_size, 1);
  assert.equal(result.config.default_request.sampling.num_inference_steps, 9);
  assert.equal(result.config.default_request.sampling.seed, 1000);
  assert.equal(result.config.default_request.sampling.num_frames, 124);
  assert.equal(result.config.default_request.sampling.fps, 24);
  assert.equal(result.config.generator.pipeline.experimental.attention_backend, "VIDEO_SPARSE_ATTN_H3");
  assert.equal(result.config.generator.pipeline.experimental.VSA_sparsity, 0.8);
  assert.equal(result.config.default_request.negative_prompt, "");
  assert.equal(result.config.default_request.output.return_frames, false);
  assert.equal(result.command.includes("examples/serving/"), false);
  assert.equal(result.recipe.evidence, "source-configured");
  const edited = resolveRecipe(input, "fasth3-v2", { values: { port: 9000, vae_offload: false, seed: 42 } });
  assert.match(edited.command, /--server.port 9000/);
  assert.match(edited.command, /--generator.engine.offload.vae false/);
  assert.match(edited.command, /--default_request.sampling.seed 42/);
  assert.match(edited.healthCommand, /:9000\/health/);
  assert.match(edited.clientCommand, /:9000\/v1\/videos\/sync/);
});

test("three recipes resolve independent defaults and expose only their task/model controls", () => {
  const input = JSON.parse(readFileSync(new URL("../../docs/assets/cookbook-serving-example.json", import.meta.url), "utf8"));
  const original = JSON.stringify(input);
  resolveRecipe(input, "fasth3-v2", { values: { seed: 3, port: 9000 }, request: { prompt: "Changed H3 prompt" } });
  for (const [id, count, steps, video] of [
    ["fasth3-v2", 4, 9, true], ["zimage-turbo", 1, 8, false], ["wan21-i2v", 2, 40, true],
  ]) {
    const result = resolveRecipe(input, id);
    assert.equal(result.config.generator.engine.num_gpus, count);
    assert.equal(result.config.generator.engine.parallelism.sp_size, count);
    assert.equal(result.config.default_request.sampling.num_inference_steps, steps);
    assert.equal(result.config.server.port, 8000);
    assert.notEqual(result.clientRequest.body.prompt, "Changed H3 prompt");
    assert.equal(result.options.some((option) => option.key === "num_frames"), video);
    assert.equal(result.options.some((option) => option.path.endsWith("VSA_sparsity")), id === "fasth3-v2");
    assert.equal(result.options.some((option) => option.key === "guidance_scale"), id === "zimage-turbo");
    assert.equal(Object.hasOwn(result.clientRequest.body, "input_reference"), id === "wan21-i2v");
  }
  const image = resolveRecipe(input, "zimage-turbo", { values: { seed: 123 } });
  assert.equal(image.clientRequest.body.seed, 123);
  assert.equal(image.clientRequest.body.guidance_scale, 0);
  assert.equal(image.clientRequest.body.size, "1024x1024");
  assert.equal(image.clientRequest.body.output_format, "png");
  assert.equal(image.config.default_request.sampling.num_frames, 1);
  const wan = resolveRecipe(input, "wan21-i2v");
  assert.equal(wan.config.default_request.sampling.num_frames, 77);
  assert.equal(wan.clientRequest.body.input_reference, "/path/to/first-frame.png");
  assert.equal(Object.hasOwn(wan.clientRequest.body, "task"), false);
  assert.equal(JSON.stringify(input), original, "switching resolutions must not mutate shared or model defaults");
});
