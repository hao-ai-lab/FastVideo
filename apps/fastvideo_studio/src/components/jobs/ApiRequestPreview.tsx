'use client';

import * as React from 'react';
import { Check, Code2, Copy, RefreshCw } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { getApiBaseUrl } from '@/lib/api';
import { requestToCurl, type ApiRequest } from '@/lib/apiRequest';

interface ApiRequestPreviewProps {
  request: ApiRequest;
}

export default function ApiRequestPreview({ request }: ApiRequestPreviewProps) {
  const [isOpen, setIsOpen] = React.useState(false);
  const [snapshot, setSnapshot] = React.useState<{ key: string; command: string } | null>(null);
  const [copyState, setCopyState] = React.useState<'idle' | 'copied' | 'error'>('idle');
  const requestKey = JSON.stringify(request);
  const isStale = snapshot !== null && snapshot.key !== requestKey;
  const codeId = React.useId();

  function refresh() {
    setSnapshot({ key: requestKey, command: requestToCurl(getApiBaseUrl(), request) });
    setCopyState('idle');
  }

  async function copy() {
    if (!snapshot || isStale) return;
    try {
      await navigator.clipboard.writeText(snapshot.command);
      setCopyState('copied');
    } catch {
      setCopyState('error');
    }
  }

  return (
    <section className="rounded-xl border border-border bg-muted/30 p-3.5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2 text-sm font-semibold">
          <Code2 aria-hidden="true" className="size-4 text-primary" />
          API request
          <span className="rounded-md bg-muted px-1.5 py-0.5 font-mono text-xs font-normal text-muted-foreground">cURL</span>
        </div>
        <Button
          type="button"
          size="sm"
          variant="ghost"
          aria-expanded={isOpen}
          aria-controls={codeId}
          onClick={() => {
            if (!isOpen) refresh();
            setIsOpen(!isOpen);
          }}
        >
          {isOpen ? 'Hide cURL' : 'Show cURL'}
        </Button>
      </div>
      <p className="mt-1 text-xs text-muted-foreground">
        {request.method === 'PATCH'
          ? 'Save this pending job with the same settings from your terminal.'
          : 'Create a pending job with these settings from your terminal, then start it from the job list.'}
      </p>
      {isOpen && snapshot && (
        <div id={codeId} className="mt-3 space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <p role="status" className={`text-xs ${isStale ? 'text-amber-600 dark:text-amber-400' : 'text-muted-foreground'}`}>
              {isStale ? 'Settings changed. Refresh to include your edits.' : 'Matches the form when last refreshed.'}
            </p>
            <div className="flex gap-2">
              <Button type="button" variant="outline" size="sm" onClick={refresh}>
                <RefreshCw aria-hidden="true" className="size-3.5" /> Refresh
              </Button>
              <Button type="button" variant="secondary" size="sm" onClick={copy} disabled={isStale}>
                {copyState === 'copied' ? <Check aria-hidden="true" className="size-3.5" /> : <Copy aria-hidden="true" className="size-3.5" />}
                {copyState === 'copied' ? 'Copied' : 'Copy cURL'}
              </Button>
            </div>
          </div>
          <pre aria-label="cURL command" tabIndex={0} className="max-h-72 overflow-auto rounded-lg border border-border bg-background p-3 font-mono text-xs leading-relaxed">
            <code>{snapshot.command}</code>
          </pre>
          <p className="text-xs text-muted-foreground">
            Uses the API URL from Settings. Uploaded media and dataset paths refer to files already on that backend.
          </p>
          {copyState === 'error' && (
            <p role="alert" className="text-xs text-destructive">Clipboard unavailable. Select the command above and copy it manually.</p>
          )}
        </div>
      )}
    </section>
  );
}
