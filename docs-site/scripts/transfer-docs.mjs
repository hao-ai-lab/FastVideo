import { modelDirectory } from "../lib/cookbook-pages.ts";
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

function localTarget(target, from, pretty = false) {
  if (/^(?:#|mailto:|data:|tel:)/.test(target)) return target;
  let url;
  try {
    url = new URL(target, `https://transfer.invalid${pretty ? `${routeFor(from)}/` : `/${from}`}`);
  } catch {
    return target;
  }
  if (url.host !== "transfer.invalid") {
    if (url.origin !== "https://haoailab.com" || !url.pathname.startsWith("/FastVideo/")) return target;
    url = new URL(url.pathname.replace(/^\/FastVideo/, "") + url.search + url.hash, "https://transfer.invalid");
  }
  const path = decodeURIComponent(url.pathname).replace(/^\//, "");
  // Links leaving docs/ name repository files, not files served by the docs host.
  if (!pretty && !target.startsWith("/") && !/^[a-z]+:/i.test(target)) {
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
        .replace(/<(?=[0-9=])/g, "&lt;")
        .replace(/(\]\()([^\s)]+)(\))/g, (_, before, url, after) => before + localTarget(url, from) + after)
        .replace(
          /((?:href|src)=["'])([^"']+)(["'])/g,
          (_, before, url, after) => before + localTarget(url, from, from.startsWith("cookbook/")) + after,
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

/** Blume owns the page title, including examples whose Source link precedes their H1. */
function extractPageTitle(body, data, deduplicate = false) {
  const normalize = (text) => text.replace(/[^\p{L}\p{N}]/gu, "").toLowerCase();
  let fence;
  let found = false;
  return body
    .split("\n")
    .map((line) => {
      const marker = /^\s*(`{3,}|~{3,})/.exec(line)?.[1];
      if (marker) {
        if (!fence) fence = marker;
        else if (marker[0] === fence[0] && marker.length >= fence.length) fence = undefined;
        return line;
      }
      if (fence || (found && !deduplicate)) return line;
      const heading = (deduplicate ? /^(#{1,2})\s+(.+)$/ : /^(#)\s+(.+)$/).exec(line);
      if (!heading) return line;
      if (heading[1].length === 2 && !data.title) return line;
      const anchor = /(?:\[#([^\]]+)\]|\{#([^}]+)\})\s*$/.exec(heading[2]);
      const embedded = /<a\s+id="[^"]*"><\/a>/g;
      const anchors = heading[2].match(embedded) ?? [];
      const title = heading[2]
        .replace(/\s*(?:\[#[^\]]+\]|\{#[^}]+\})\s*$/, "")
        .replace(embedded, "")
        .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
        .replace(/[`*_]/g, "")
        .trim();
      if (data.title && normalize(data.title) !== normalize(title)) return line;
      if (heading[1].length === 1) {
        data.title ??= title;
        found = true;
      }
      if (anchor) anchors.push(`<a id="${anchor[1] ?? anchor[2]}"></a>`);
      return anchors.join("\n");
    })
    .join("\n");
}

export async function transferDocs() {
  const localPython = join(site, ".venv/bin/python");
  const python = process.env.DOCS_PYTHON || (existsSync(localPython) ? localPython : "python3");
  execFileSync(python, [join(site, "scripts/prepare-python-docs.py")], { cwd: repo, stdio: "inherit" });
  await rm(content, { recursive: true, force: true });
  await rm(publicDir, { recursive: true, force: true });
  await mkdir(content, { recursive: true });
  const ignored = new Set(["js", "tests", "node_modules", "__pycache__"]);
  await cp(source, content, {
    recursive: true,
    filter: async (path) => {
      const rel = relative(source, path);
      if (ignored.has(rel.split(sep)[0]) || rel === "README.md") return false;
      return (await stat(path)).isDirectory() || /\.mdx?$/.test(path);
    },
  });
  // Native callout directives and islands compile as MDX, while source links keep .md filenames.
  for (const file of await files(content)) {
    if (!file.endsWith(".md")) continue;
    const text = await readFile(file, "utf8");
    if (/^:::(?:note|tip|warning|danger|info|success)\b/m.test(text) || text.includes("<RecipeFamily ")) {
      await writeFile(file.replace(/\.md$/, ".mdx"), text);
      await rm(file);
    }
  }
  const oldHeadings = JSON.parse(await readFile(join(generated, "old-headings.json"), "utf8"));
  for (const path of await files(content)) {
    if (!/\.mdx?$/.test(path)) continue;
    const rel = relative(content, path).replaceAll(sep, "/");
    const route = routeFor(rel);
    const { data, body } = splitPage(await readFile(path, "utf8"));
    delete data.noindex;
    data.slug = route.replace(/^\//, "") || "/";
    const pinned = extractPageTitle(
      pinHeadings(body, oldHeadings[route.slice(1) || "."], path.endsWith(".mdx")),
      data,
      rel.includes("/examples/") || rel.startsWith("examples/"),
    );
    // GitHub's buttons remain ordinary links; no third-party script is needed in the new shell.
    let converted = rewriteLinks(pinned, rel).replace(/<script\b[^>]*>[\s\S]*?<\/script>/g, "");
    converted = converted.replace(/^(\s*`{3,})slurm\b/gm, "$1bash");
    if (path.endsWith(".mdx"))
      converted = converted.replace(
        /<style>([\s\S]*?)<\/style>/g,
        (_, css) => `<style>{${JSON.stringify(css)}}</style>`,
      );
    {
      const aliases = (oldHeadings[route.slice(1)] ?? []).filter(
        (heading) =>
          !converted.includes(`id="${heading.id}"`) &&
          !converted.includes(`[#${heading.id}]`) &&
          !converted.includes(`{#${heading.id}}`),
      );
      converted += "\n" + aliases.map((heading) => `<a id="${heading.id}"></a>`).join("\n");
    }
    await writeFile(path, `---\n${stringify(data)}---\n${converted}`);
  }
  // API HTML is extracted directly from Python source with Griffe.
  await rm(join(content, "getting_started/v1_api.md"), { force: true });
  await rm(join(content, "getting_started/v1_api.mdx"), { force: true });
  await cp(join(generated, "python-reference"), content, { recursive: true });
  const originalHome = splitPage(await readFile(join(content, "index.md"), "utf8")).body;
  const section = originalHome.indexOf("## Key Features");
  if (section < 0) throw new Error("The homepage source needs an updated section boundary.");
  await mkdir(join(content, "_snippets"), { recursive: true });
  await mkdir(join(content, "_snippets/clients"), { recursive: true });
  for (const [file, language] of [
    ["video.py", "python"],
    ["video.mjs", "javascript"],
    ["video.sh", "bash"],
  ]) {
    const code = await readFile(join(repo, "examples/serving/clients", file), "utf8");
    await writeFile(join(content, "_snippets/clients", `${file}.md`), `\`\`\`${language}\n${code}\n\`\`\`\n`);
  }
  const asMdx = (body) =>
    body
      .replace(/\{#([^}]+)\}/g, "[#$1]")
      .replace(/<img\b([^>]*?)\/?\s*>/g, "<img$1 />")
      .replace(/ style="[^"]*"/g, "");
  const description = originalHome
    .slice(0, section)
    .split(/\n\s*\n/)
    .find((paragraph) => paragraph.trim().startsWith("FastVideo is an inference"));
  if (!description) throw new Error("The homepage description paragraph needs updating.");
  await writeFile(join(content, "_snippets/home-description.mdx"), asMdx(description.trim()));
  const documentation = originalHome.indexOf("## Documentation", section);
  if (documentation < 0) throw new Error("The homepage documentation section boundary needs updating.");
  await writeFile(join(content, "_snippets/home-details.mdx"), asMdx(originalHome.slice(section, documentation)));

  // The README owns announcements and the community list; the welcome page renders a short news digest.
  const readme = await readFile(join(repo, "README.md"), "utf8");
  const news = readme.match(/^## NEWS\n([\s\S]*?)(?=^##|^###)/m)?.[1];
  const projects = readme.match(/^## Awesome work using FastVideo or our research projects\n([\s\S]*?)(?=^## )/m)?.[1];
  if (!news || !projects) throw new Error("The README news or project section boundary needs updating.");
  const recent = [...news.matchAll(/^- `([\d/]+)`: (.+)$/gm)].slice(0, 2).map(([, date, text]) => {
    const links = [...text.matchAll(/\[([^\]]+)\]\(([^)]+)\)/g)];
    const headline = text
      .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
      .split(/:\s|\.\s/)[0]
      .replace(/\.$/, "");
    const href = links.at(-1)?.[2] ?? "https://github.com/hao-ai-lab/FastVideo#news";
    const iso = date.replaceAll("/", "-");
    const label = new Intl.DateTimeFormat("en", {
      month: "short",
      day: "numeric",
      year: "numeric",
      timeZone: "UTC",
    }).format(new Date(iso));
    return `<li><time dateTime="${iso}">${label}</time><a href={${JSON.stringify(href)}}>{${JSON.stringify(headline)}}</a></li>`;
  });
  await writeFile(
    join(content, "_snippets/home-news.mdx"),
    `<section className="home-news not-prose" aria-labelledby="recent-news">\n<div className="home-news-heading"><h2 id="recent-news">Recent news</h2><a className="home-news-more" href="https://github.com/hao-ai-lab/FastVideo#news">All announcements →</a></div>\n<ul>\n${recent.join("\n")}\n</ul>\n</section>\n`,
  );
  await writeFile(join(content, "_snippets/home-projects.mdx"), asMdx(projects.trim()));
  await rm(join(content, "index.md"));
  await rm(join(content, "cookbook/index.md"), { force: true });
  await rm(join(content, "cookbook/index.mdx"), { force: true });
  await cp(join(site, "content"), content, { recursive: true });
  // Model-specific inference pages come from the maintained recipe inventory.
  for (const model of modelDirectory.filter((model) => !model.configurable)) {
    const file = join(content, `${model.path.slice(1)}.mdx`);
    await mkdir(dirname(file), { recursive: true });
    await writeFile(
      file,
      `---\ntitle: ${JSON.stringify(model.title)}\nmode: frame\n---\n\n<Cookbook modelId="${model.id}" />\n`,
    );
  }

  await cp(join(source, "assets"), join(publicDir, "assets"), {
    recursive: true,
    filter: (path) => !path.endsWith(".md"),
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
  const navigation = JSON.parse(await readFile(join(site, "navigation.json"), "utf8"));
  function checkNavigation(items) {
    for (const item of items) {
      if (typeof item === "string") {
        if (!routes.has(item)) throw new Error(`Navigation target is missing: ${item}`);
      } else {
        if (item.href && !routes.has(item.href)) throw new Error(`Navigation target is missing: ${item.href}`);
        if (item.items) checkNavigation(item.items);
      }
    }
  }
  checkNavigation(navigation);
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
  function addModelPages(items) {
    for (const item of items) {
      const entries = modelDirectory.filter((model) => !model.configurable && model.familyPath === item.href);
      if (entries.length && !/Family guide|Recipes and setup/.test(item.label)) {
        const guide = { label: "Family guide", href: item.href };
        delete item.href;
        item.display = "group";
        item.collapsed = true;
        item.items = [...entries.map((model) => ({ label: model.title, href: model.path })), guide];
      } else if (item.items) {
        const guide = item.items.find((child) => /Family guide|Recipes and setup/.test(child.label));
        if (guide) {
          const extras = modelDirectory.filter((model) => !model.configurable && model.familyPath === guide.href);
          item.items.splice(
            item.items.indexOf(guide),
            0,
            ...extras.map((model) => ({ label: model.title, href: model.path })),
          );
        }
        addModelPages(item.items);
      }
    }
  }
  addModelPages(navigation);
  await writeFile(join(generated, "navigation.json"), JSON.stringify(navigation));
  await writeFile(join(generated, "routes.json"), JSON.stringify([...routes].sort()));
  console.log(
    `Transferred ${routes.size} documentation routes. Canonical guide routes and cookbook runbooks are published together.`,
  );
}
