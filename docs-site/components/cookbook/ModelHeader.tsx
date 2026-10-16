import type { ReactNode } from "react";
import { sitePath, type DirectoryModel } from "@/lib/cookbook-pages";

export default function ModelHeader({ model, actions }: { model: DirectoryModel; actions?: ReactNode }) {
  return (
    <>
      <a className="recipe-back" href={sitePath("/cookbook")}>
        ← All recipes
      </a>
      <header className="recipe-model-header">
        <img src={sitePath(`/assets/logos/${model.logo}.webp`)} alt="" width="48" height="48" />
        <div>
          <p className="recipe-publisher">{model.family}</p>
          <h1>{model.title}</h1>
          <p className="recipe-model-task">{model.tags.join(" · ")}</p>
          {!model.model.startsWith("./") && (
            <a href={`https://huggingface.co/${model.model}`}>View on Hugging Face ↗</a>
          )}
        </div>
        <div className="recipe-model-actions">{actions}</div>
      </header>
    </>
  );
}
