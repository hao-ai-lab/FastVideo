import { defineConfig } from "@playwright/test";

const externalBase = process.env.PLAYWRIGHT_BASE_URL;
const localBase = "http://127.0.0.1:4327";

export default defineConfig({
  testDir: "./tests",
  testMatch: "**/*.spec.ts",
  use: { baseURL: externalBase ?? localBase, trace: "retain-on-failure" },
  webServer: externalBase
    ? undefined
    : {
        command: "npm run preview -- --host 127.0.0.1 --port 4327",
        url: `${localBase}/FastVideo/cookbook`,
        reuseExistingServer: !process.env.CI,
      },
});
