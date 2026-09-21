'use client';

import * as React from 'react';
import { Check, Code2, Copy, RefreshCw } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { getApiBaseUrl } from '@/lib/api';
import { getRequestPathVariables, requestToCurl, type ApiRequest } from '@/lib/apiRequest';
import CurlCode from './CurlCode';

interface ApiRequestPreviewProps {
  id?: string;
  request: ApiRequest;
}

export default function ApiRequestPreview({ id, request }: ApiRequestPreviewProps) {
  const [snapshot, setSnapshot] = React.useState(() => ({
    key: JSON.stringify(request),
    command: requestToCurl(getApiBaseUrl(), request),
    usesPathVariables: getRequestPathVariables(request).length > 0,
  }));
  const [copyState, setCopyState] = React.useState<'idle' | 'copied' | 'error'>('idle');
  const requestKey = JSON.stringify(request);
  const isStale = snapshot.key !== requestKey;

  function refresh() {
    setSnapshot({
      key: requestKey,
      command: requestToCurl(getApiBaseUrl(), request),
      usesPathVariables: getRequestPathVariables(request).length > 0,
    });
    setCopyState('idle');
  }

  async function copy() {
    if (isStale) return;
    try {
      await navigator.clipboard.writeText(snapshot.command);
      setCopyState('copied');
    } catch {
      setCopyState('error');
    }
  }

  return (
    <section id={id} aria-label="API request example" className="min-w-0 space-y-3 rounded-xl border border-border bg-muted/30 p-3.5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2 text-sm font-semibold">
          <Code2 aria-hidden="true" className="size-4 text-primary" />
          cURL example
        </div>
        <div className="ml-auto flex items-center gap-2">
          <Button type="button" variant="outline" size="sm" onClick={refresh}>
            <RefreshCw aria-hidden="true" className="size-3.5" /> Refresh
          </Button>
          <Button
            type="button"
            variant="secondary"
            size="icon-sm"
            aria-label={copyState === 'copied' ? 'Copied cURL' : 'Copy cURL'}
            title={isStale ? 'Refresh to copy your latest settings' : copyState === 'copied' ? 'Copied cURL' : 'Copy cURL'}
            onClick={copy}
            disabled={isStale}
          >
            {copyState === 'copied' ? <Check aria-hidden="true" className="size-4" /> : <Copy aria-hidden="true" className="size-4" />}
          </Button>
        </div>
      </div>
      <p className="mt-1 text-xs text-muted-foreground">
        {request.method === 'PATCH'
          ? 'Save this pending job with the same settings from your terminal.'
          : 'Create a pending job with these settings from your terminal, then start it from the job list.'}
      </p>
      <p role="status" className={`text-xs ${isStale ? 'text-amber-600 dark:text-amber-400' : 'text-muted-foreground'}`}>
        {isStale ? 'Settings changed. Refresh to include your edits.' : 'Matches the form when last refreshed.'}
      </p>
      <pre aria-label="cURL command" tabIndex={0} className="max-h-72 min-w-0 max-w-full overflow-auto rounded-lg border border-border bg-background p-3 font-mono text-xs leading-relaxed">
        <CurlCode command={snapshot.command} />
      </pre>
      <p className="text-xs text-muted-foreground">
        Uses the API URL from Settings. Uploaded media and dataset paths refer to files already on that backend.
        {snapshot.usesPathVariables && ' Edit the path exports above to change inputs. This example requires jq to encode paths safely as JSON. DATA_PATH and VALIDATION_DATASET_PATH may also contain a saved dataset ID.'}
      </p>
      {copyState === 'error' && (
        <p role="alert" className="text-xs text-destructive">Clipboard unavailable. Select the command above and copy it manually.</p>
      )}
    </section>
  );
}
