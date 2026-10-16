import { useId, useState } from "react";
import { families } from "@/lib/fast-models";

export const client = "load";
const local = (path: string) => `${import.meta.env.BASE_URL.replace(/\/$/, "")}${path}`;

export default function FastModels() {
  const [family, setFamily] = useState("All models");
  const [query, setQuery] = useState("");
  const id = useId();
  const visible = families
    .map((group) => ({
      ...group,
      models: group.models.filter(
        (model) =>
          (family === "All models" || family === group.name) &&
          `${group.name} ${group.summary} ${model.id}`.toLowerCase().includes(query.trim().toLowerCase()),
      ),
    }))
    .filter((group) => group.models.length);
  const count = visible.reduce((total, group) => total + group.models.length, 0);
  return (
    <div className="models-page not-prose">
      <section className="model-families" aria-label="Model families">
        {families.map((group) => (
          <article className="model-family" key={group.name}>
            <a href={group.collection} className="model-family-visual" aria-label={`${group.name} collection`}>
              <span aria-hidden="true">{group.mark}</span>
              <img src={local("/icon.svg")} alt="" width="24" height="24" />
            </a>
            <div className="model-family-body">
              <h2>{group.name}</h2>
              <p>{group.summary}</p>
              <p className="model-family-detail">{group.detail}</p>
              <div className="model-family-links">
                <a href={group.collection}>
                  View collection <span aria-hidden="true">↗</span>
                </a>
                <a href={local(group.guide)}>
                  Setup guide <span aria-hidden="true">→</span>
                </a>
              </div>
            </div>
          </article>
        ))}
      </section>
      <section className="model-catalog" aria-labelledby="checkpoint-catalog">
        <header className="model-catalog-heading">
          <div>
            <h2 id="checkpoint-catalog">Browse checkpoints</h2>
            <p>Find the export your runtime needs.</p>
          </div>
          <a href="https://huggingface.co/collections/FastVideo/fastvideo-fasth3">
            Earlier FastH3 releases <span aria-hidden="true">↗</span>
          </a>
        </header>
        <div className="model-catalog-controls">
          <div className="model-filters" role="group" aria-label="Filter by model family">
            {["All models", ...families.map((group) => group.name)].map((name) => (
              <button key={name} type="button" aria-pressed={family === name} onClick={() => setFamily(name)}>
                {name}
              </button>
            ))}
          </div>
          <div className="model-search">
            <label htmlFor={id}>Search checkpoints</label>
            <input
              id={id}
              type="search"
              placeholder="Name, precision, or runtime…"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
          </div>
        </div>
        <p className="model-result-count" role="status">
          {count} {count === 1 ? "checkpoint" : "checkpoints"}
        </p>
        {visible.map((group) => (
          <section className="model-checkpoints" key={group.name} aria-labelledby={`catalog-${group.name}`}>
            <h3 id={`catalog-${group.name}`}>{group.name}</h3>
            <ul>
              {group.models.map((model) => (
                <li key={model.id}>
                  <div>
                    <a href={model.url}>{model.id.replace(/^FastVideo-/, "").replace(/-Diffusers$/, "")}</a>
                    <code>FastVideo/{model.id}</code>
                  </div>
                  <a className="model-checkpoint-link" href={model.url} aria-label={`Open ${model.id} on Hugging Face`}>
                    Hugging Face <span aria-hidden="true">↗</span>
                  </a>
                </li>
              ))}
            </ul>
          </section>
        ))}
        {!count && (
          <div className="model-empty">
            <p>No checkpoints match these filters.</p>
            <button
              type="button"
              onClick={() => {
                setQuery("");
                setFamily("All models");
              }}
            >
              Clear filters
            </button>
          </div>
        )}
      </section>
    </div>
  );
}
