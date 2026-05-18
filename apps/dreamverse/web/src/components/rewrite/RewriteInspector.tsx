'use client';

import React, { useMemo } from 'react';

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
import { Badge } from '@/components/ui/badge';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';

interface RewriteInspectorProps {
  currentPromptWindowPrompts?: string[];
  promptEvents?: Record<string, any>[];
  rewritingSeedPrompts?: boolean;
  rewriteWindowMode?: boolean;
}

function isRewriteEvent(event: Record<string, any>): boolean {
  const status = String(event?.status || '').trim();
  const source = String(event?.source || '').trim();
  return (
    source === 'llm_rewrite' ||
    source === 'user_rewrite' ||
    status.startsWith('rewrite_')
  );
}

function trimText(value: any): string {
  return typeof value === 'string' ? value.trim() : '';
}

function toPreview(text: any): string {
  const normalized = trimText(text);
  if (!normalized) {
    return '';
  }
  return normalized.length > 180
    ? `${normalized.slice(0, 180).trimEnd()}...`
    : normalized;
}

export default function RewriteInspector({
  currentPromptWindowPrompts = [],
  promptEvents = [],
  rewritingSeedPrompts = false,
  rewriteWindowMode = false,
}: RewriteInspectorProps) {
  const rewriteEvents = useMemo(
    () => promptEvents.filter(isRewriteEvent),
    [promptEvents],
  );

  const latestRewriteRequest = useMemo(
    () =>
      rewriteEvents.find(
        (event) =>
          event.status === 'rewrite_requested' &&
          event.source === 'user_rewrite',
      ) || null,
    [rewriteEvents],
  );

  const latestRewriteResult = useMemo(
    () =>
      rewriteEvents.find(
        (event) =>
          event.status === 'rewrite_ready' ||
          event.status === 'rewrite_fallback' ||
          event.status === 'rewrite_error',
      ) || null,
    [rewriteEvents],
  );

  const latestRewriteRawOutput = useMemo(
    () =>
      rewriteEvents.find(
        (event) => event.status === 'rewrite_raw_output',
      ) || null,
    [rewriteEvents],
  );

  const hasRewriteActivity = rewriteEvents.length > 0;

  const statusLabel = rewritingSeedPrompts
    ? 'Running'
    : rewriteWindowMode
      ? 'Armed'
      : 'Idle';

  const statusVariant = rewritingSeedPrompts
    ? 'warning'
    : rewriteWindowMode
      ? 'default'
      : 'secondary';

  return (
    <aside aria-label="Rewrite inspector" className="h-full">
      <Card className="h-full overflow-hidden">
        <CardHeader className="gap-4 pb-4">
          <div className="flex items-start justify-between gap-3">
            <div>
              <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-sky-200/70">
                Rewrite
              </p>
              <CardTitle className="mt-1 text-xl">Rewrite inspector</CardTitle>
            </div>
            <Badge variant={statusVariant}>{statusLabel}</Badge>
          </div>
          <CardDescription>
            Inspect the active prompt window and the most recent rewrite
            activity without leaving the workspace.
          </CardDescription>
        </CardHeader>

        <CardContent className="space-y-5">
          <section className="space-y-3">
            <div className="flex items-center justify-between gap-3">
              <div>
                <h3 className="text-sm font-semibold text-foreground">
                  Current window snapshot
                </h3>
                <p className="text-sm text-muted-foreground">
                  Prompts currently driving generation.
                </p>
              </div>
              <Badge variant="secondary">
                {currentPromptWindowPrompts.length}
              </Badge>
            </div>

            {currentPromptWindowPrompts.length === 0 ? (
              <div className="rounded-2xl border border-dashed border-border bg-secondary px-4 py-6 text-sm text-muted-foreground">
                No prompts are in the active window yet.
              </div>
            ) : (
              <ScrollArea className="max-h-80 pr-4">
                <div className="space-y-3">
                  {currentPromptWindowPrompts.map((prompt, index) => (
                    <div
                      key={index}
                      className="rounded-2xl border border-border bg-secondary p-4"
                    >
                      <div className="mb-2 flex items-center justify-between gap-2">
                        <span className="text-[11px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">
                          Prompt {index + 1}
                        </span>
                      </div>
                      <p className="text-sm leading-6 text-foreground">
                        {prompt}
                      </p>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            )}
          </section>

          <Separator />

          <section className="space-y-3">
            <div className="flex items-center justify-between gap-3">
              <div>
                <h3 className="text-sm font-semibold text-foreground">
                  Latest rewrite
                </h3>
                <p className="text-sm text-muted-foreground">
                  Recent requests, results, and raw model output.
                </p>
              </div>
              <Badge variant="secondary">
                {hasRewriteActivity ? rewriteEvents.length : 0} events
              </Badge>
            </div>

            {!hasRewriteActivity ? (
              <div className="rounded-2xl border border-dashed border-border bg-secondary px-4 py-6 text-sm text-muted-foreground">
                Rewrite activity will appear here once the window is rewritten.
              </div>
            ) : (
              <div className="space-y-4">
                {latestRewriteRequest && (
                  <div className="rounded-2xl border border-border bg-secondary p-4">
                    <div className="mb-2 flex items-center justify-between gap-3">
                      <span className="text-[11px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">
                        Instruction
                      </span>
                      <Badge variant="outline">Request</Badge>
                    </div>
                    <p className="text-sm leading-6 text-foreground">
                      {toPreview(latestRewriteRequest.text) ||
                        'Rewrite current prompt window'}
                    </p>
                  </div>
                )}

                {latestRewriteResult && (
                  <div className="rounded-2xl border border-border bg-secondary p-4">
                    <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                      <span className="text-[11px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">
                        Result
                      </span>
                      <div className="flex flex-wrap gap-2">
                        <Badge
                          variant={
                            latestRewriteResult.status === 'rewrite_error'
                              ? 'destructive'
                              : latestRewriteResult.status ===
                                  'rewrite_fallback'
                                ? 'warning'
                                : 'success'
                          }
                        >
                          {latestRewriteResult.status}
                        </Badge>
                        {latestRewriteResult.model && (
                          <Badge variant="outline">
                            {latestRewriteResult.model}
                          </Badge>
                        )}
                        {Number.isFinite(latestRewriteResult.latencyMs) && (
                          <Badge variant="secondary">
                            {Math.round(latestRewriteResult.latencyMs)}ms
                          </Badge>
                        )}
                      </div>
                    </div>
                    <p className="text-sm leading-6 text-foreground">
                      {latestRewriteResult.text}
                    </p>
                  </div>
                )}

                {latestRewriteRawOutput && (
                  <Accordion type="single" collapsible>
                    <AccordionItem value="raw-output">
                      <AccordionTrigger>Raw model output</AccordionTrigger>
                      <AccordionContent>
                        <div className="rounded-2xl bg-card p-4 font-mono text-xs leading-6 text-muted-foreground">
                          {latestRewriteRawOutput.text}
                        </div>
                      </AccordionContent>
                    </AccordionItem>
                  </Accordion>
                )}
              </div>
            )}
          </section>
        </CardContent>
      </Card>
    </aside>
  );
}
