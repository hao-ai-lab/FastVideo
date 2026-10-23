import js from "@eslint/js";
import globals from "globals";

const nodeFiles = ["docs/js/build-cookbook-*.mjs", "docs/tests/*.test.mjs", "docs/*config.mjs"];
const browserFiles = ["docs/js/cookbook-*.mjs"];

export default [
  {
    files: [...nodeFiles, ...browserFiles],
    rules: js.configs.recommended.rules,
    languageOptions: { ecmaVersion: "latest", sourceType: "module" },
  },
  { files: nodeFiles, languageOptions: { globals: globals.node } },
  { files: browserFiles, languageOptions: { globals: globals.browser } },
];
