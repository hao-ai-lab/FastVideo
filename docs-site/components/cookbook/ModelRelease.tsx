import ModelHeader from "./ModelHeader";
import ComponentPlan from "./ComponentPlan";
import { sitePath, type DirectoryModel } from "@/lib/cookbook-pages";

export default function ModelRelease({ model }: { model: DirectoryModel }) {
  return (
    <div className="cookbook not-prose recipe-detail">
      <ModelHeader model={model} />
      <div className="release-intro">
        <p>FastH3 Trim is the smaller FastH3 model, previously called CompactH3.</p>
        <p>A maintained serving recipe is being prepared. Use the published checkpoint instructions for setup.</p>
      </div>
      <section className="recipe-matrix" aria-label="Model releases">
        <div className="matrix-row">
          <h2>Weights</h2>
          <div className="matrix-row-content">
            <div className="option-chips">
              {[
                ["BF16", "FastVideo-FastH3-Trim-8-Step"],
                ["FP8", "FastVideo-FastH3-Trim-8-Step-FP8"],
                ["NVFP4", "FastVideo-FastH3-Trim-8-Step-NVFP4"],
              ].map(([name, repo]) => (
                <a className="option-chip-static" key={repo} href={`https://huggingface.co/FastVideo/${repo}`}>
                  {name} ↗
                </a>
              ))}
            </div>
          </div>
        </div>
        <div className="matrix-row">
          <h2>Components</h2>
          <div className="matrix-row-content">
            <ComponentPlan />
          </div>
        </div>
      </section>
      <div className="recipe-source-links">
        <a href={sitePath("/inference/fasth3-distilled")}>FastH3 checkpoint guide →</a>
        <a href="https://huggingface.co/collections/FastVideo/fasth3-v2">Release collection ↗</a>
      </div>
    </div>
  );
}
