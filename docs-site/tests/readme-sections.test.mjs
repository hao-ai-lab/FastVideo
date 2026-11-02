import assert from "node:assert/strict";
import { test } from "node:test";
import { readReadmeSections } from "../lib/readme.mjs";

const legacy = `# FastVideo

## NEWS
- \`2026/10/06\`: FastH3 V2 runs on a single consumer machine.
- \`2026/09/15\`: Release FastH3 8-Step V2.

## Awesome work using FastVideo or our research projects
- [Some project](https://example.com)

## Community
- Slack
`;

const rewritten = `# FastVideo

## Overview
FastVideo is a framework.

## Recent updates
- **2026-10-06:** FastH3 V2 runs on a single consumer machine.
- **2026-09-15:** Release FastH3 8-Step V2.

## Projects using FastVideo
- [Some project](https://example.com)

## Community and contributing
- Slack
`;

test("reads the legacy README headings and dates", () => {
  const { news, projects } = readReadmeSections(legacy);
  assert.equal(news.length, 2);
  assert.deepEqual(
    news.map((item) => item.date),
    ["2026/10/06", "2026/09/15"],
  );
  assert.match(news[0].text, /^FastH3 V2/);
  assert.match(projects, /Some project/);
});

test("reads a rewritten README with sentence case headings and bold dates", () => {
  const { news, projects } = readReadmeSections(rewritten);
  assert.equal(news?.length, 2);
  assert.deepEqual(
    news.map((item) => item.date),
    ["2026/10/06", "2026/09/15"],
  );
  assert.doesNotMatch(news[0].text, /\*\*/);
  assert.match(projects, /Some project/);
  assert.doesNotMatch(projects, /Slack/);
});

test("stops at the next heading and ignores other sections", () => {
  const { news } = readReadmeSections(legacy);
  assert.doesNotMatch(news[1].text, /Awesome work/);
});

test("returns null when a section is missing", () => {
  assert.deepEqual(readReadmeSections("# FastVideo\n\n## Overview\n"), { news: null, projects: null });
});
