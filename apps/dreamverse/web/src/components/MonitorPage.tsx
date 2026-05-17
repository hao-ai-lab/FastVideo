'use client';

import React, { useState, useEffect, useRef } from 'react';

import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

const POLL_INTERVAL_MS = 15000;

interface Replica {
  url: string;
  healthy: boolean;
  active_sessions?: number;
  pending_sessions?: number;
  max_available_sessions?: number;
  prompt_provider_success_counts?: Record<string, number>;
}

function formatTimestamp(value: string): string {
  if (!value) return '';
  try {
    return new Date(value).toLocaleString();
  } catch {
    return '';
  }
}

function formatProviderSuccessCounts(
  counts: Record<string, number> | undefined,
): string {
  const normalized = counts || {};
  const cerebrasIfm = normalized.cerebras_ifm ?? 0;
  const cerebras = normalized.cerebras ?? 0;
  const groq = normalized.groq ?? 0;
  return `IFM ${cerebrasIfm} / Cerebras ${cerebras} / Groq ${groq}`;
}

export default function MonitorPage() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [replicas, setReplicas] = useState<Replica[]>([]);
  const [lastUpdated, setLastUpdated] = useState('');
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    async function loadReplicaSessions() {
      try {
        const response = await fetch('/router/replicas/sessions');
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(
            payload.detail || 'Failed to fetch replica sessions',
          );
        }
        setReplicas(
          Array.isArray(payload.replicas) ? payload.replicas : [],
        );
        setError('');
        setLastUpdated(new Date().toISOString());
      } catch (err: any) {
        setError(err?.message || String(err));
      } finally {
        setLoading(false);
      }
    }

    loadReplicaSessions();
    timerRef.current = setInterval(loadReplicaSessions, POLL_INTERVAL_MS);

    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
  }, []);

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-6xl flex-col gap-4 px-4 py-8 text-foreground">
      <div className="space-y-2">
        <h1 className="text-3xl font-semibold text-foreground">
          Replica Session Monitor
        </h1>
        <div className="flex flex-wrap gap-2 text-sm text-muted-foreground">
          <Badge variant="secondary">poll interval: 15s</Badge>
        {lastUpdated && (
            <Badge variant="outline">
              last updated: {formatTimestamp(lastUpdated)}
            </Badge>
        )}
        </div>
      </div>

      {loading ? (
        <Card>
          <CardContent className="p-6 text-sm text-muted-foreground">
            Loading monitor data...
          </CardContent>
        </Card>
      ) : error ? (
        <Card className="border-rose-500/30 bg-rose-950/45">
          <CardContent className="p-6 text-sm text-rose-100">
            {error}
          </CardContent>
        </Card>
      ) : (
        <Card className="overflow-hidden">
          <CardHeader className="border-b border-border pb-4">
            <CardTitle className="text-xl">Replica capacity</CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <div className="overflow-x-auto">
              <table className="w-full min-w-[760px] border-collapse text-left text-sm">
                <thead className="bg-secondary text-muted-foreground">
                  <tr>
                    <th className="border-b border-border px-4 py-3 font-semibold">
                      URL
                    </th>
                    <th className="border-b border-border px-4 py-3 font-semibold">
                      Healthy
                    </th>
                    <th className="border-b border-border px-4 py-3 font-semibold">
                      Active WS Sessions
                    </th>
                    <th className="border-b border-border px-4 py-3 font-semibold">
                      Pending Sessions
                    </th>
                    <th className="border-b border-border px-4 py-3 font-semibold">
                      Max Available Sessions
                    </th>
                    <th className="border-b border-border px-4 py-3 font-semibold">
                      Prompt API Successes
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {replicas.map((replica, index) => (
                    <tr
                      key={replica.url || index}
                      className="border-b border-border last:border-b-0"
                    >
                      <td className="px-4 py-3 text-foreground">{replica.url}</td>
                      <td className="px-4 py-3">
                        <Badge
                          variant={replica.healthy ? 'success' : 'destructive'}
                        >
                          {replica.healthy ? 'yes' : 'no'}
                        </Badge>
                      </td>
                      <td className="px-4 py-3 text-foreground">
                        {replica.active_sessions ?? '-'}
                      </td>
                      <td className="px-4 py-3 text-foreground">
                        {replica.pending_sessions ?? '-'}
                      </td>
                      <td className="px-4 py-3 text-foreground">
                        {replica.max_available_sessions ?? '-'}
                      </td>
                      <td className="px-4 py-3 text-foreground">
                        {formatProviderSuccessCounts(
                          replica.prompt_provider_success_counts,
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      )}
    </main>
  );
}
