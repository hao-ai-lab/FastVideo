import { modelDirectory, sitePath } from "@/lib/cookbook-pages";

/** Family runbooks link to the same model pages as the cookbook catalog. */
export default function RecipeFamily({ family }: { family: string }) {
  const models = modelDirectory.filter(
    (model) =>
      model.recipes.some((recipe) => recipe.family === family) ||
      (family === "minimax_h3" && (model.family === "FastH3" || model.familyPath === "/cookbook/minimax-h3")),
  );
  return (
    <div className="recipe-directory not-prose">
      <a className="recipe-back" href={sitePath("/cookbook")}>
        ← All models
      </a>
      <div className="recipe-card-grid">
        {models.map((model) => (
          <a className="recipe-card" key={model.id} href={sitePath(model.path)}>
            <div className="recipe-card-top">
              <img src={sitePath(`/assets/logos/${model.logo}.webp`)} alt="" width="40" height="40" />
              <span className="recipe-card-kicker">{model.provider}</span>
            </div>
            <h2>{model.title}</h2>
            <div className="recipe-tags">
              <span>{model.category}</span>
              {model.tags.slice(0, 2).map((tag) => (
                <span key={tag}>{tag}</span>
              ))}
            </div>
            <footer>
              <span>
                {model.platforms.map((platform) => (platform === "Apple Silicon" ? "MLX" : platform)).join(" · ")}
              </span>
              <span>{model.configurable ? "Configure →" : "View recipe →"}</span>
            </footer>
          </a>
        ))}
      </div>
    </div>
  );
}
