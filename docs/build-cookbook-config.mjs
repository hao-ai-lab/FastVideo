/** Build static cookbook catalogs from authored YAML. No FastVideo or Python imports. */
import { existsSync, lstatSync, mkdirSync, readFileSync, readdirSync, realpathSync, renameSync, unlinkSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, extname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { parseDocument, stringify } from "yaml";

const require = createRequire(import.meta.url);
const { createValidator } = require("./assets/cookbook-validator.js");
const { resolveConfig } = require("./assets/cookbook-config.js");
export const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const ID = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
const reserved = new Set(["__proto__", "constructor", "prototype"]);
const annotations = new Set(["default", "examples", "enum", "const"]);
const protectedPaths = new Set([
  "generator.model_path", "generator.pipeline.workload_type", "generator.pipeline.preset",
  "generator.pipeline.preset_version", "generator.pipeline.components.vae_weights",
  "default_request.output.save_video", "default_request.output.return_frames", "default_request.output.output_path",
]);
const object = (value) => value !== null && typeof value === "object" && !Array.isArray(value);
const clone = (value) => structuredClone(value);
const requireValue = (condition, message) => { if (!condition) throw new Error(message); };
const text = (value) => typeof value === "string" && value.trim().length > 0;

/** Object keys merge recursively; arrays and scalar values replace, without mutating inputs. */
export function deepMerge(base, overlay) {
  if (!object(base) || !object(overlay)) return clone(overlay);
  return Object.fromEntries([...new Set([...Object.keys(base), ...Object.keys(overlay)])].map((key) =>
    [key, Object.hasOwn(overlay, key) ? deepMerge(base[key], overlay[key]) : clone(base[key])]));
}

function readYaml(path) {
  const document = parseDocument(readFileSync(path, "utf8"), { uniqueKeys: true, intAsBigInt: true, merge: true });
  requireValue(!document.errors.length, `${path}: ${document.errors.map((error) => error.message).join("; ")}`);
  const active = new Set();
  function jsonValue(value) {
    if (typeof value === "bigint") {
      requireValue(value >= BigInt(Number.MIN_SAFE_INTEGER) && value <= BigInt(Number.MAX_SAFE_INTEGER),
        `${path}: integer exceeds browser-safe range`);
      return Number(value);
    }
    if (typeof value === "number") {
      requireValue(Number.isFinite(value), `${path}: non-finite number`);
      requireValue(!Number.isInteger(value) || Number.isSafeInteger(value), `${path}: integer exceeds browser-safe range`);
    }
    if (value === null || typeof value !== "object") return value;
    requireValue(!active.has(value), `${path}: cyclic YAML aliases are not JSON configuration`);
    active.add(value);
    const result = Array.isArray(value) ? value.map(jsonValue) :
      Object.fromEntries(Object.entries(value).map(([key, child]) => [key, jsonValue(child)]));
    active.delete(value);
    return result;
  }
  return jsonValue(document.toJS({ maxAliasCount: 100 }));
}

function keys(value, required, optional, context) {
  requireValue(object(value) && required.every((key) => Object.hasOwn(value, key)) &&
    Object.keys(value).every((key) => [...required, ...optional].includes(key)),
  `${context}: expected ${required.join(", ")}; optional ${optional.join(", ")}`);
}

function safePath(path) {
  requireValue(typeof path === "string" &&
    /^(generator|server|default_request)(\.[A-Za-z_][A-Za-z0-9_]*)*$/.test(path) &&
    path.includes(".") && !path.split(".").some((part) => reserved.has(part)),
  `Invalid configuration path: ${path}`);
}

function declaredTypes(schema) {
  if (typeof schema.type === "string") return [schema.type];
  if (Array.isArray(schema.type)) return [...schema.type].sort();
  const alternatives = schema.anyOf || schema.oneOf;
  if (!Array.isArray(alternatives)) return null;
  const branches = alternatives.map(declaredTypes);
  return branches.every(Boolean) ? [...new Set(branches.flat())].sort() : null;
}

/** Merge metadata separately from control selection. A partial override never enables a field. */
export function mergeOptions(base, overrides, context = "options") {
  requireValue(object(base) && object(overrides), `${context}: options must be a path-to-schema mapping`);
  for (const [path, schema] of Object.entries(overrides)) {
    safePath(path);
    requireValue(object(schema), `${context}: ${path} must contain a schema mapping`);
  }
  const result = deepMerge(base, overrides);
  for (const [path, schema] of Object.entries(result)) {
    const previous = base[path] && declaredTypes(base[path]);
    if (previous) requireValue(JSON.stringify(previous) === JSON.stringify(declaredTypes(schema)),
      `${context}: ${path} cannot change its declared type`);
  }
  return result;
}

/** Validate self-contained fragments, including bounds/defaults that schema syntax alone cannot check. */
function validateSchema(schema, context) {
  requireValue(object(schema) && declaredTypes(schema), `${context}: a complete schema needs an explicit type`);
  function visit(node) {
    if (Array.isArray(node)) { node.forEach(visit); return; }
    if (!object(node)) return;
    requireValue(!Object.hasOwn(node, "$ref"), `${context}: $ref is unsupported; use a self-contained schema`);
    for (const [minimum, maximum] of [["minLength", "maxLength"], ["minItems", "maxItems"],
      ["minProperties", "maxProperties"], ["minContains", "maxContains"]]) {
      if (minimum in node && maximum in node) requireValue(node[minimum] <= node[maximum],
        `${context}: ${minimum} exceeds ${maximum}`);
    }
    const lower = Math.max(node.minimum ?? -Infinity, node.exclusiveMinimum ?? -Infinity);
    const upper = Math.min(node.maximum ?? Infinity, node.exclusiveMaximum ?? Infinity);
    requireValue(lower <= upper && !(lower === upper &&
      (node.exclusiveMinimum === lower || node.exclusiveMaximum === upper)), `${context}: minimum exceeds maximum`);
    for (const [key, child] of Object.entries(node)) {
      if (annotations.has(key)) continue;
      if (["properties", "$defs", "patternProperties", "dependentSchemas"].includes(key)) Object.values(child).forEach(visit);
      else visit(child);
    }
    if (Object.hasOwn(node, "default")) {
      const validate = createValidator(node);
      requireValue(validate(node.default), `${context}: default does not satisfy its schema`);
    }
    if (Array.isArray(node.enum)) {
      const withoutEnum = { ...node }; delete withoutEnum.enum;
      const validate = createValidator(withoutEnum);
      requireValue(node.enum.every((value) => validate(value)), `${context}: enum value does not satisfy its schema`);
    }

  }
  try { visit(schema); createValidator(schema); }
  catch (failure) { throw new Error(`${context}: ${failure.message}`); }
}

function validateOptions(options, context) {
  for (const [path, schema] of Object.entries(options)) { safePath(path); validateSchema(schema, `${context}: ${path}`); }
}

function within(root, source, folder, extensions) {
  requireValue(text(source) && !isAbsolute(source), `Expected a repository-relative path under ${folder}`);
  const path = realpathSync(resolve(root, source)), base = realpathSync(join(root, folder));
  const child = relative(base, path);
  requireValue(child && child !== ".." && !child.startsWith(`..${sep}`) && !isAbsolute(child) &&
    extensions.includes(extname(path)) && lstatSync(path).isFile(), `Expected an existing ${extensions.join("/")} file under ${folder}: ${source}`);
  return path;
}

function guideLink(value, root) {
  if (value === undefined || value === null) return null;
  const path = within(root, value, "docs", [".md"]);
  const parts = relative(realpathSync(join(root, "docs")), path).split(sep);
  const leaf = parts.pop();
  if (leaf !== "index.md") parts.push(leaf.slice(0, -3));
  return { url: `../../${parts.map(encodeURIComponent).join("/")}${parts.length ? "/" : ""}` };
}

/** Match canonical field paths with '*' and optional leading '!' exclusion. */
function optionPattern(selector) {
  requireValue(typeof selector === "string" && selector.length > 0, `Invalid option pattern: ${selector}`);
  const exclude = selector.startsWith("!"), pattern = exclude ? selector.slice(1) : selector;
  requireValue(!pattern.includes("**") && pattern.split(".").every((part) =>
    /^[A-Za-z_*][A-Za-z0-9_*]*$/.test(part) && !reserved.has(part)), `Invalid option pattern: ${selector}`);
  const expression = pattern.split("*").map((part) => part.replace(/\./g, "\\.")).join(".*");
  return { exclude, match: new RegExp(`^${expression}$`) };
}

/** Apply selection rules before checking the final editable fields. */
function expandOptions(options, selectors, runtime) {
  requireValue(Array.isArray(selectors), "options must be an ordered list of paths or patterns");
  const available = Object.keys(options), selected = new Set();
  for (const selector of selectors) {
    const { exclude, match } = optionPattern(selector);
    const paths = available.filter((path) => match.test(path));
    requireValue(paths.length, `Unknown option pattern: ${selector}`);
    for (const path of paths) { if (exclude) selected.delete(path); else selected.add(path); }
  }
  const forbidden = [...protectedPaths, ...(runtime.interface === "websocket" ? ["server.output_dir", "server.served_model_name"] : [])];
  const seen = new Set(), controls = [];
  for (const path of selected) {
    requireValue(!forbidden.some((item) => item === path || item.startsWith(`${path}.`) || path.startsWith(`${item}.`)),
      `Protected configuration option: ${path}`);
    requireValue(![...seen].some((item) => item.startsWith(`${path}.`) || path.startsWith(`${item}.`)),
      `Overlapping options: ${path}`);
    seen.add(path);
    controls.push({ path, schema: clone(options[path]) });
  }
  return controls;
}

export function buildCatalogs({ root = ROOT, recipesDir = join(root, "docs/cookbook/recipes"),
  catalogPath = join(root, "docs/cookbook/options.yaml") } = {}) {
  const declarations = readYaml(catalogPath);
  keys(declarations, ["options", "runtimes"], [], "Option catalog");
  requireValue(object(declarations.options) && object(declarations.runtimes), "options and runtimes must be mappings");
  const runtimes = new Map();
  for (const [id, definition] of Object.entries(declarations.runtimes)) {
    keys(definition, ["metadata", "workloads", "options"], [], `Runtime ${id}`);
    const runtime = definition.metadata;
    requireValue(object(runtime) && runtime.id === id && text(runtime.label) && text(runtime.backend) &&
      ["rest", "websocket"].includes(runtime.interface) && Array.isArray(runtime.launch_argv) &&
      runtime.launch_argv.length && runtime.launch_argv.every(text) && object(runtime.server_defaults) &&
      text(runtime.server_defaults.host) && Number.isInteger(runtime.server_defaults.port), `Invalid runtime metadata: ${id}`);
    requireValue(Array.isArray(definition.workloads) && definition.workloads.length &&
      definition.workloads.every((workload) => ["t2v", "i2v"].includes(workload)), `Invalid workloads for ${id}`);
    const options = mergeOptions(declarations.options, definition.options, `Runtime ${id}`);
    validateOptions(options, `Runtime ${id}`);
    runtimes.set(id, { runtime, workloads: definition.workloads, options });
  }
  const catalogs = [], modelKeys = new Set(), modelOwners = new Map();
  const paths = readdirSync(recipesDir).filter((name) => /\.ya?ml$/.test(name)).sort();
  requireValue(paths.length, `No serving recipe manifests in ${recipesDir}`);
  for (const name of paths) {
    const modelKey = name.replace(/\.ya?ml$/, "");
    requireValue(ID.test(modelKey) && !modelKeys.has(modelKey), `Invalid or duplicate model key: ${modelKey}`);
    modelKeys.add(modelKey);
    const manifest = readYaml(join(recipesDir, name));
    keys(manifest, ["title", "deployments"], ["summary", "guide"], name);
    requireValue(text(manifest.title) && object(manifest.deployments) && Object.keys(manifest.deployments).length,
      `${name}: title and at least one deployment are required`);
    let modelId;
    for (const [deploymentKey, deployment] of Object.entries(manifest.deployments)) {
      const id = `${modelKey}/${deploymentKey}`;
      requireValue(ID.test(deploymentKey), `Invalid deployment key: ${id}`);
      keys(deployment, ["runtime", "workload", "defaults", "options"], ["label", "summary", "guide", "env", "requirements", "overrides"], id);
      requireValue(runtimes.has(deployment.runtime), `${id}: unknown runtime ${deployment.runtime}`);
      const { runtime, workloads, options } = runtimes.get(deployment.runtime);
      requireValue(workloads.includes(deployment.workload), `${id}: unsupported workload ${deployment.workload}`);
      const baseline = readYaml(within(root, deployment.defaults, "examples/serving", [".yaml", ".yml"]));
      requireValue(object(baseline) && text(baseline.generator?.model_path), `${id}: baseline needs generator.model_path`);
      const currentModel = baseline.generator.model_path;
      requireValue(modelId === undefined || modelId === currentModel, `${id}: deployments must use the same model_path`);
      requireValue(!modelOwners.has(currentModel) || modelOwners.get(currentModel) === modelKey,
        `${id}: duplicate model identity; add the deployment to ${modelOwners.get(currentModel)}.yaml`);
      modelId = currentModel; modelOwners.set(modelId, modelKey);
      const streaming = baseline.streaming !== undefined && baseline.streaming !== null;
      requireValue(!streaming || object(baseline.streaming), `${id}: streaming must be a configuration mapping`);
      requireValue((runtime.interface === "websocket") === streaming, `${id}: streaming block disagrees with runtime interface`);
      if (baseline.runtime !== undefined) requireValue(baseline.runtime === runtime.backend, `${id}: baseline runtime disagrees with backend`);
      if (baseline.generator.pipeline?.workload_type !== undefined) requireValue(
        baseline.generator.pipeline.workload_type === deployment.workload, `${id}: baseline workload disagrees with deployment`);
      const env = deployment.env ?? {}, requirements = deployment.requirements ?? [];
      requireValue(object(env) && Object.entries(env).every(([key, value]) =>
        /^[A-Za-z_][A-Za-z0-9_]*$/.test(key) && typeof value === "string" && !value.includes("\0")), `${id}: invalid environment mapping`);
      requireValue(Array.isArray(requirements) && requirements.every(text), `${id}: requirements must be text notes`);
      for (const value of [manifest.summary, deployment.summary]) requireValue(value === undefined || typeof value === "string", `${id}: summary must be text`);
      requireValue(deployment.label === undefined || text(deployment.label), `${id}: label must be nonempty text`);
      const merged = mergeOptions(options, deployment.overrides ?? {}, id);
      validateOptions(merged, id);
      const catalog = {
        id, title: manifest.title, summary: deployment.summary ?? manifest.summary ?? "", source_config: deployment.defaults,
        model: { id: modelId, key: modelKey, title: manifest.title }, deployment: { id: deploymentKey, label: deployment.label ?? runtime.label },
        workload: deployment.workload, runtime: clone(runtime), env: clone(env), base_config: baseline,
        controls: expandOptions(merged, deployment.options, runtime), requirements: clone(requirements),
        guide: guideLink(Object.hasOwn(deployment, "guide") ? deployment.guide : manifest.guide, root),
      };
      // Share the browser's authored-field and topology checks. Never materialize schema defaults in YAML.
      try { resolveConfig(catalog); } catch (failure) { throw new Error(`${id}: ${failure.message}`); }
      catalogs.push(catalog);
    }
  }
  return catalogs;
}

/** Render one validated deployment catalog as YAML without writing generated files. */
export function previewCatalog(recipeId, options = {}) {
  requireValue(text(recipeId), "Preview requires a model/deployment ID");
  const catalogs = buildCatalogs(options), catalog = catalogs.find((item) => item.id === recipeId);
  const available = catalogs.map((item) => item.id).join(", ");
  requireValue(catalog, `Unknown cookbook deployment: ${recipeId}. Available: ${available}`);
  return stringify(catalog, { aliasDuplicateObjects: false });
}

function writeDocument(path, value) {
  const document = `${JSON.stringify(value, null, 2)}\n`;
  if (existsSync(path) && readFileSync(path, "utf8") === document) return;
  writeFileSync(`${path}.tmp`, document); renameSync(`${path}.tmp`, path);
}

export function exportCatalogs({ outputDir = join(ROOT, "docs/assets/cookbook-config"), ...options } = {}) {
  const catalogs = buildCatalogs(options), index = { models: [] }, models = new Map();
  const expected = new Set();
  const directories = [outputDir, join(outputDir, "recipes"), ...catalogs.map((catalog) => join(outputDir, "recipes", catalog.model.key))];
  requireValue(directories.every((path) => !existsSync(path) || !lstatSync(path).isSymbolicLink()), "Generated directories must not be symlinks");
  for (const catalog of catalogs) {
    const url = `recipes/${catalog.id}.json`;
    expected.add(url);
    const directory = join(outputDir, "recipes", catalog.model.key);
    mkdirSync(directory, { recursive: true });
    writeDocument(join(outputDir, url), catalog);
    if (!models.has(catalog.model.key)) {
      const model = { id: catalog.model.key, title: catalog.model.title, model_id: catalog.model.id, deployments: [] };
      models.set(model.id, model); index.models.push(model);
    }
    models.get(catalog.model.key).deployments.push({ id: catalog.id, label: catalog.deployment.label,
      workload: catalog.workload, runtime: catalog.runtime.id, catalog_url: url });
  }
  writeDocument(join(outputDir, "index.json"), index);
  for (const model of readdirSync(join(outputDir, "recipes"), { withFileTypes: true })) {
    if (!model.isDirectory() || !ID.test(model.name)) continue;
    for (const file of readdirSync(join(outputDir, "recipes", model.name))) {
      const url = `recipes/${model.name}/${file}`;
      if (file.endsWith(".json") && ID.test(file.slice(0, -5)) && !expected.has(url)) unlinkSync(join(outputDir, url));
    }
  }
  return index;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    const options = {};
    const flags = {
      "--output-dir": "outputDir", "--recipes-dir": "recipesDir", "--catalog": "catalogPath", "--preview": "previewRecipe",
    };
    for (let index = 2; index < process.argv.length; index += 2) {
      const flag = process.argv[index], value = process.argv[index + 1];
      requireValue(Object.hasOwn(flags, flag) && value && !value.startsWith("--"),
        "Use --preview MODEL/DEPLOYMENT, --output-dir PATH, --recipes-dir PATH or --catalog PATH");
      options[flags[flag]] = flag === "--preview" ? value : resolve(value);
    }
    if (Object.hasOwn(options, "previewRecipe")) {
      requireValue(!Object.hasOwn(options, "outputDir"),
        "--output-dir cannot be used with --preview; redirect stdout to save YAML");
      const { previewRecipe, ...builderOptions } = options;
      process.stdout.write(previewCatalog(previewRecipe, builderOptions));
    } else {
      exportCatalogs(options);
      console.log(join(options.outputDir || join(ROOT, "docs/assets/cookbook-config"), "index.json"));
    }
  } catch (failure) { console.error(failure.message); process.exitCode = 1; }
}
