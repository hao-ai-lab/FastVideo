'use client';

import * as React from 'react';

import { CATEGORY_STYLE } from '@/components/jobs/H3TagPalette';
import { dialogueTree, type DialogueNode } from '@/lib/h3DialogueEstimate';
import { H3_TAG_CATEGORIES } from '@/lib/h3Tags';
import { cn } from '@/lib/utils';

const CATEGORY_BY_TAG_NAME: Record<string, string> = Object.fromEntries(
  H3_TAG_CATEGORIES.flatMap((c) => c.tags.map((t) => [t.open.replace(/[<>]/g, ''), c.id])),
);

function renderPreviewNodes(nodes: DialogueNode[], keyPrefix = ''): React.ReactNode[] {
  return nodes.map((n, i) => {
    const key = `${keyPrefix}${i}`;
    if (n.t === 'text') return <span key={key}>{n.v}</span>;
    if (n.t === 'point') {
      const cat = CATEGORY_BY_TAG_NAME[n.name] ?? 'body';
      const style = CATEGORY_STYLE[cat] ?? CATEGORY_STYLE.body;
      return (
        <span
          key={key}
          className={cn(
            'mx-0.5 inline-block rounded border px-1.5 py-0.5 font-mono text-[11px] leading-none',
            style.chip,
          )}
        >
          {n.name}
        </span>
      );
    }
    const inner = renderPreviewNodes(n.children, `${key}-`);
    if (n.name === 'i') return <strong key={key} className="italic">{inner}</strong>;
    if (n.name === 'whisper') return <span key={key} className="text-[0.92em] tracking-wide opacity-60">{inner}</span>;
    if (n.name === 'humming') return <em key={key} className="underline decoration-wavy decoration-emerald-500 underline-offset-4">{inner}</em>;
    return <span key={key}>{inner}</span>;
  });
}

/** Styled "performance" rendering of speech tags: chips for point tags, styling for wrapping ones. */
export function H3DialoguePreview({ text, className }: { text: string; className?: string }) {
  return <div className={className}>{renderPreviewNodes(dialogueTree(text))}</div>;
}
