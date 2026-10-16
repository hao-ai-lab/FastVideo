/** Rebuild the Blume view from canonical Markdown and the existing Python reference renderer. */
import { execFileSync } from "node:child_process";
import { existsSync, statSync } from "node:fs";
import { cp, mkdir, readdir, readFile, rm, stat, writeFile } from "node:fs/promises";
import { dirname, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { parse, stringify } from "yaml";

const site = fileURLToPath(new URL("../", import.meta.url));
const repo = resolve(site, "..");
const source = join(repo, "docs");
const generated = join(site, ".generated");
const content = join(generated, "content");
const publicDir = join(site, "public");
const attachments = new Set();
const legacyOrigin = "https://haoailab.com/FastVideo";
const routeFor = (file) =>
  "/" +
  file
    .replace(/\\/g, "/")
    .replace(/\.(md|mdx)$/, "")
    .replace(/(^|\/)index$/, "")
    .replace(/\/$/, "");

async function files(root) {
  const result = [];
  for (const entry of await readdir(root, { withFileTypes: true })) {
    const path = join(root, entry.name);
    if (entry.isDirectory()) result.push(...(await files(path)));
    else result.push(path);
  }
  return result;
}

function splitPage(text) {
  const match = /^---\n([\s\S]*?)\n---\n/.exec(text);
  return match ? { data: parse(match[1]), body: text.slice(match[0].length) } : { data: {}, body: text };
}

function localTarget(target, from) {
  if (/^(?:#|mailto:|data:|tel:)/.test(target)) return target;
  let url;
  try {
    url = new URL(target, `https://transfer.invalid/${from}`);
  } catch {
    return target;
  }
  if (url.host !== "transfer.invalid") {
    if (url.origin !== "https://haoailab.com" || !url.pathname.startsWith("/FastVideo/")) return target;
    url = new URL(url.pathname.replace(/^\/FastVideo/, "") + url.search + url.hash, "https://transfer.invalid");
  }
  const path = decodeURIComponent(url.pathname).replace(/^\//, "");
  if (path.startsWith("cookbook/") || path === "cookbook") {
    const route = routeFor(path);
    return ["/cookbook", "/cookbook/"].includes(route)
      ? `/cookbook${url.search}${url.hash}`
      : `${legacyOrigin}${route}${url.search}${url.hash}`;
  }
  // Links leaving docs/ name repository files, not files served by the docs host.
  if (!target.startsWith("/") && !/^[a-z]+:/i.test(target)) {
    const actual = resolve(source, dirname(from), decodeURIComponent(target.split(/[?#]/)[0]));
    if (!actual.startsWith(source + sep) && existsSync(actual)) {
      return `https://github.com/hao-ai-lab/FastVideo/blob/main/${relative(repo, actual).replaceAll(sep, "/")}${url.hash}`;
    }
  }
  if (existsSync(join(source, path)) && statSync(join(source, path)).isFile() && !/\.mdx?$/.test(path))
    attachments.add(path);
  if (/\.mdx?$/.test(path)) return `${routeFor(path)}${url.search}${url.hash}`;
  return `/${path}${url.search}${url.hash}`;
}

function rewriteLinks(body, from) {
  let fence;
  return body
    .split("\n")
    .map((line) => {
      const marker = /^\s*(`{3,}|~{3,})/.exec(line)?.[1];
      if (marker) {
        if (!fence) fence = marker;
        else if (marker[0] === fence[0] && marker.length >= fence.length) fence = undefined;
        return line;
      }
      if (fence) return line;
      return line
        .replace(/\{\s*\.md-button[^}]*\}/g, "")
        .replace(/<(?=[0-9=])/g, "&lt;")
        .replace(/(\]\()([^\s)]+)(\))/g, (_, before, url, after) => before + localTarget(url, from) + after)
        .replace(
          /((?:href|src)=["'])([^"']+)(["'])/g,
          (_, before, url, after) => before + localTarget(url, from) + after,
        );
    })
    .join("\n");
}

function pinHeadings(body, old, mdx) {
  const normalize = (text) =>
    text
      .replace(/<[^>]*>/g, "")
      .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
      .replace(/[^\p{L}\p{N}]/gu, "")
      .toLowerCase();
  const candidates = [...(old ?? [])];
  let fence;
  let text = body
    .split("\n")
    .map((line) => {
      const marker = /^\s*(`{3,}|~{3,})/.exec(line)?.[1];
      if (marker) {
        if (!fence) fence = marker;
        else if (marker[0] === fence[0] && marker.length >= fence.length) fence = undefined;
        return line;
      }
      if (fence) return line;
      const heading = /^(#{1,6})\s+(.+?)\s*$/.exec(line);
      if (!heading) return line;
      const clean = heading[2].replace(/\s*(?:\[#.*?\]|\{#.*?\})\s*$/, "");
      const index = candidates.findIndex(
        (item) => item.level === heading[1].length && normalize(item.text.replace(/¶$/, "")) === normalize(clean),
      );
      if (index < 0) return line;
      const [original] = candidates.splice(index, 1);
      if (original.id.includes("__")) return `${heading[1]} <a id="${original.id}"></a>${clean}`;
      return `${heading[1]} ${clean} ${mdx ? `[#${original.id}]` : `{#${original.id}}`}`;
    })
    .join("\n");
  const first = candidates.find((item) => item.level === 1);
  if (first && !text.includes(`id="${first.id}"`)) text = `<a id="${first.id}"></a>\n\n${text}`;
  return text;
}

export async function transferDocs() {
  const localPython = join(site, ".venv/bin/python");
  const python = process.env.DOCS_PYTHON || (existsSync(localPython) ? localPython : "python3");
  execFileSync(python, [join(site, "scripts/prepare-python-docs.py")], { cwd: repo, stdio: "inherit" });
  await rm(content, { recursive: true, force: true });
  await mkdir(content, { recursive: true });
  const ignored = new Set(["cookbook", "js", "tests", "node_modules", "__pycache__"]);
  await cp(source, content, {
    recursive: true,
    filter: async (path) => {
      const rel = relative(source, path);
      if (ignored.has(rel.split(sep)[0]) || rel === "README.md") return false;
      return (await stat(path)).isDirectory() || /\.mdx?$/.test(path);
    },
  });
  const blumePackage = dirname(fileURLToPath(import.meta.resolve("blume/package.json")));
  const report = execFileSync(
    process.execPath,
    [
      join(blumePackage, "skills/blume-migrate/scripts/mkdocs-codemod.mjs"),
      "--write",
      "--config",
      join(repo, "mkdocs.yml"),
      content,
    ],
    { cwd: repo, encoding: "utf8" },
  );
  await writeFile(join(generated, "conversion-report.log"), report);
  const oldHeadings = JSON.parse(await readFile(join(generated, "old-headings.json"), "utf8"));
  for (const path of await files(content)) {
    if (!/\.mdx?$/.test(path)) continue;
    const rel = relative(content, path).replaceAll(sep, "/");
    const route = routeFor(rel);
    const { data, body } = splitPage(await readFile(path, "utf8"));
    data.noindex = true;
    data.slug = route.replace(/^\//, "") || "/";
    const pinned = pinHeadings(body, oldHeadings[route.slice(1) || "."], path.endsWith(".mdx"));
    // GitHub's buttons remain ordinary links; no third-party script is needed in the new shell.
    let converted = rewriteLinks(pinned, rel).replace(/<script\b[^>]*>[\s\S]*?<\/script>/g, "");
    converted = converted.replace(/^(\s*`{3,})slurm\b/gm, "$1bash");
    if (path.endsWith(".mdx"))
      converted = converted.replace(
        /<style>([\s\S]*?)<\/style>/g,
        (_, css) => `<style>{${JSON.stringify(css)}}</style>`,
      );
    await writeFile(path, `---\n${stringify(data)}---\n${converted}`);
  }
  // API HTML is already rendered by mkdocstrings, including aliases and cross references.
  await rm(join(content, "getting_started/v1_api.md"), { force: true });
  await rm(join(content, "getting_started/v1_api.mdx"), { force: true });
  await cp(join(generated, "python-reference"), content, { recursive: true });
  const originalHome = splitPage(await readFile(join(content, "index.md"), "utf8")).body;
  const section = originalHome.indexOf("## Key Features");
  if (section < 0) throw new Error("The homepage source needs an updated section boundary.");
  await mkdir(join(content, "_snippets"), { recursive: true });
  const asMdx = (body) =>
    body
      .replace(/\{#([^}]+)\}/g, "[#$1]")
      .replace(/<img\b([^>]*?)\/?\s*>/g, "<img$1 />")
      .replace(/ style="[^"]*"/g, "");
  await writeFile(
    join(content, "_snippets/home-introduction.mdx"),
    asMdx(originalHome.slice(0, section).replace(/<a id="welcome-to-fastvideo"><\/a>/g, "")),
  );
  await writeFile(join(content, "_snippets/home-details.mdx"), asMdx(originalHome.slice(section)));
  await rm(join(content, "index.md"));
  await cp(join(site, "content"), content, { recursive: true });
  await cp(join(source, "assets"), join(publicDir, "assets"), {
    recursive: true,
    filter: (path) => !path.endsWith(".md") && !/cookbook(?:-recipes|-serving)?\.(?:js|json)$/.test(path),
  });

  for (const path of attachments) {
    await mkdir(dirname(join(publicDir, path)), { recursive: true });
    await cp(join(source, path), join(publicDir, path));
  }
  const routes = new Set(
    (await files(content))
      .filter(
        (path) =>
          /\.mdx?$/.test(path) &&
          !relative(content, path)
            .split(sep)
            .some((part) => /^[_.]/.test(part)),
      )
      .map((path) => routeFor(relative(content, path))),
  );
  function navEntries(entries) {
    return entries.flatMap((entry) =>
      Object.entries(entry).map(([label, value]) => {
        if (label === "Cookbook") return { label, items: ["/cookbook"] };
        if (Array.isArray(value)) return { label, items: navEntries(value) };
        const route = routeFor(value);
        if (!routes.has(route)) throw new Error(`Navigation target is missing: ${value}`);
        return { label, href: route };
      }),
    );
  }
  const navigation = navEntries(JSON.parse(await readFile(join(generated, "mkdocs-nav.json"), "utf8")));
  // Keep generated modules discoverable without putting all 880 entries in every page's sidebar.
  for (const path of await files(join(content, "api"))) {
    if (!path.endsWith(".md")) continue;
    const { data, body } = splitPage(await readFile(path, "utf8"));
    const folder = dirname(path);
    const children = (await readdir(folder, { withFileTypes: true })).filter((item) => item.isDirectory());
    const links = children
      .map(
        (item) =>
          `<li><a href="${routeFor(relative(content, join(folder, item.name, "index.md")))}">${item.name}</a></li>`,
      )
      .join("\n");
    await writeFile(
      path,
      `---\n${stringify(data)}---\n${body.replaceAll("{", "&#123;").replaceAll("}", "&#125;")}${links ? `\n<h2>Submodules</h2>\n<ul>\n${links}\n</ul>\n` : ""}`,
    );
  }
  const start = navigation.find((item) => item.label === "Getting Started");
  if (start) {
    start.label = "Get started";
    start.icon = "rocket";
    const quickStart = start.items.find((item) => item.href === "/getting_started/quick_start");
    start.items = [quickStart, ...start.items.filter((item) => item !== quickStart)];
  }
  for (const group of navigation) {
    if (group.label === "Home") group.label = "Welcome";
    if (["Training", "Distillation", "Attention", "Utilities", "Design", "Developer Guide"].includes(group.label)) {
      group.display = "group";
      group.collapsed = true;
    }
  }
  const apiGroup = navigation.find((item) => item.label === "API Reference");
  if (apiGroup) apiGroup.items.push({ label: "Python package", href: "/api/fastvideo" });
  await writeFile(join(generated, "navigation.json"), JSON.stringify(navigation));
  await writeFile(join(generated, "routes.json"), JSON.stringify([...routes].sort()));
  console.log(`Transferred ${routes.size} documentation routes. Cookbook design sources are unchanged.`);
}
