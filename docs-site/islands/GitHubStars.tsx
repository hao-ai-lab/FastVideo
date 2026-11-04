import { useEffect, useState } from "react";

export const client = "load";

const repository = "https://github.com/hao-ai-lab/FastVideo";
const endpoint = "https://api.github.com/repos/hao-ai-lab/FastVideo";
const compact = new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 });
const exact = new Intl.NumberFormat("en");

export default function GitHubStars() {
  const [stars, setStars] = useState<number | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    async function refresh() {
      try {
        const response = await fetch(endpoint, {
          headers: { Accept: "application/vnd.github+json" },
          signal: AbortSignal.any([controller.signal, AbortSignal.timeout(5000)]),
        });
        if (!response.ok) return;
        const data = await response.json();
        if (Number.isSafeInteger(data.stargazers_count) && data.stargazers_count >= 0 && !controller.signal.aborted) {
          setStars(data.stargazers_count);
        }
      } catch {
        // Keep the link usable when offline or when the public API is rate limited.
      }
    }
    void refresh();
    const timer = setInterval(
      () => {
        if (document.visibilityState === "visible") void refresh();
      },
      5 * 60 * 1000,
    );
    return () => {
      controller.abort();
      clearInterval(timer);
    };
  }, []);

  const label = stars === null ? "Star FastVideo on GitHub" : `FastVideo on GitHub, ${exact.format(stars)} stars`;
  return (
    <a className="home-github-stars" href={repository} aria-label={label} title={label}>
      <svg width="17" height="17" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38v-1.49c-2.23.48-2.7-1.08-2.7-1.08-.36-.92-.89-1.17-.89-1.17-.73-.5.06-.49.06-.49.8.06 1.22.82 1.22.82.71 1.22 1.87.87 2.33.67.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.96 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82A7.65 7.65 0 0 1 8 3.72c.68 0 1.36.09 2 .27 1.53-1.03 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.28.82 2.15 0 3.08-1.87 3.75-3.65 3.95.29.25.54.74.54 1.49v2.21c0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z" />
      </svg>
      <svg className="home-star" width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
        <path d="m12 2.5 2.94 5.96 6.58.96-4.76 4.64 1.12 6.55L12 17.52l-5.88 3.09 1.12-6.55L2.48 9.42l6.58-.96L12 2.5Z" />
      </svg>
      <span>{stars === null ? "Star" : compact.format(stars).toLowerCase()}</span>
    </a>
  );
}
