import { stringify } from "yaml";
import type { Result, Value } from "./cookbook";

export function pythonSource(
  source: string,
  controls: { value: string | number | boolean; spans?: number[][] }[],
  edits: Record<number, string | number | boolean> = {},
) {
  const replacements = controls.flatMap((field, index) =>
    (field.spans ?? []).map(([start, end]) => {
      const value = Object.hasOwn(edits, index) ? edits[index] : field.value;
      return { start, end, text: typeof value === "boolean" ? (value ? "True" : "False") : JSON.stringify(value) };
    }),
  );
  for (const replacement of replacements.sort((a, b) => b.start - a.start))
    source = source.slice(0, replacement.start) + replacement.text + source.slice(replacement.end);
  return source;
}

/** Quoted heredocs preserve prompts literally, including shell substitutions. */
export function configAndLaunch(yaml: string, command: string) {
  return fileAndLaunch("config.yaml", yaml, command);
}

export function fileAndLaunch(filename: string, content: string, command: string) {
  let delimiter = "FASTVIDEO_CONFIG";
  while (content.split("\n").includes(delimiter)) delimiter += "_";
  return `(\nset -e\ncat > ${filename} <<'${delimiter}'\n${content.trimEnd()}\n${delimiter}\n${command}\n)`;
}

export const sdkCommand = `python - <<'FASTVIDEO_SDK'
import yaml
from fastvideo import VideoGenerator

with open("config.yaml") as file:
    config = yaml.safe_load(file)
generator = VideoGenerator.from_config(config["generator"])
try:
    generator.generate(config["request"])
finally:
    generator.shutdown()
FASTVIDEO_SDK`;

/** The CUDA CLI uses the same generator with request instead of server defaults. */
export function inferenceOutput(result: Result) {
  const defaults = (result.config.default_request ?? {}) as Record<string, Value>;
  const output = (defaults.output ?? {}) as Record<string, Value>;
  return {
    yaml: stringify({
      generator: result.config.generator,
      request: {
        ...defaults,
        prompt: defaults.prompt ?? "A river flowing through a peaceful forest",
        output: { ...output, save_video: true, return_frames: false, output_path: "outputs/cookbook/" },
      },
    }),
    command: result.command.replace(
      /\bfastvideo serve --config config\.yaml$/,
      "fastvideo generate --config config.yaml",
    ),
  };
}
