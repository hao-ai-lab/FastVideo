const NEWS_HEADINGS = ["news", "recent updates", "what's new", "latest news", "announcements"];

const PROJECT_HEADINGS = [
  "awesome work using fastvideo or our research projects",
  "projects using fastvideo",
  "built on fastvideo",
  "awesome work",
];

function normalizeHeading(line) {
  return line
    .replace(/^#{2,3}\s+/, "")
    .replace(/[`*_]/g, "")
    .replace(/[^\p{L}\p{N}\s'’-]/gu, "")
    .replace(/\s+/g, " ")
    .trim()
    .toLowerCase();
}

function sectionBody(readme, names) {
  const lines = readme.split("\n");
  const start = lines.findIndex((line) => /^##\s+/.test(line) && names.includes(normalizeHeading(line)));
  if (start < 0) return null;
  const rest = lines.slice(start + 1);
  const end = rest.findIndex((line) => /^##\s+/.test(line));
  return (end < 0 ? rest : rest.slice(0, end)).join("\n").trim();
}

const BULLET = /^-\s+(?:`(\d{4})[/-](\d{2})[/-](\d{2})`:|\*\*(\d{4})[/-](\d{2})[/-](\d{2}):?\*\*:?)\s*(.+)$/;

export function readReadmeSections(readme) {
  const news = sectionBody(readme, NEWS_HEADINGS);
  const projects = sectionBody(readme, PROJECT_HEADINGS);
  if (news === null || projects === null) return { news: null, projects: null };
  const items = [];
  for (const line of news.split("\n")) {
    const match = BULLET.exec(line.trim());
    if (!match) continue;
    const [, y1, m1, d1, y2, m2, d2, text] = match;
    items.push({ date: `${y1 ?? y2}/${m1 ?? m2}/${d1 ?? d2}`, text: text.trim() });
  }
  return { news: items, projects };
}

export const README_HEADINGS = { news: NEWS_HEADINGS, projects: PROJECT_HEADINGS };
