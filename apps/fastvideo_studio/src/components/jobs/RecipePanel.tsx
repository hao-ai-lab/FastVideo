'use client';

import * as React from 'react';

import {
  recipeRows,
  type ModelHardware,
  type ModelRecipe,
} from '@/lib/modelCapabilities';

export interface RecipePanelProps {
  recipe: ModelRecipe;
  hardware?: ModelHardware | null;
  numGpus: number;
}

/**
 * Read-only view of a model's locked sampling recipe. These values come from
 * the checkpoint's fastvideo_inference.json and replace the generic option
 * controls, which would otherwise send defaults the checkpoint rejects.
 */
export default function RecipePanel({
  recipe,
  hardware,
  numGpus,
}: RecipePanelProps) {
  return (
    <section
      aria-labelledby="modal-recipe-heading"
      className="flex flex-col gap-2 rounded-md border border-border bg-secondary/40 p-3"
    >
      <h3
        id="modal-recipe-heading"
        className="text-sm font-medium text-foreground"
      >
        Recipe (locked to the checkpoint)
      </h3>
      <dl className="grid grid-cols-[max-content_1fr] gap-x-3 gap-y-1 text-xs">
        {recipeRows(recipe, numGpus).map((row) => (
          <React.Fragment key={row.label}>
            <dt className="text-muted-foreground">{row.label}</dt>
            <dd className="m-0 break-all font-mono text-foreground">
              {row.value}
            </dd>
          </React.Fragment>
        ))}
      </dl>
      {hardware?.supported === false && (
        <p role="alert" className="text-xs text-destructive">
          Not runnable on this server: {hardware.reason} A job will fail before
          any weights load.
        </p>
      )}
      {hardware?.supported === null && hardware.reason && (
        <p className="text-xs text-muted-foreground">
          Hardware not checked: {hardware.reason}
        </p>
      )}
    </section>
  );
}
