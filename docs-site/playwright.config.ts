import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  use: { baseURL: "http://127.0.0.1:4327", trace: "retain-on-failure" },
  webServer: {
    command: "npm run preview -- --host 127.0.0.1 --port 4327",
    url: "http://127.0.0.1:4327/FastVideo/cookbook",
    reuseExistingServer: !process.env.CI,
  },
});
