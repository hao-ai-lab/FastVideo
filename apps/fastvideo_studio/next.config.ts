import path from 'node:path';
import { fileURLToPath } from 'node:url';
import type { NextConfig } from 'next';

const configDir = path.dirname(fileURLToPath(import.meta.url));

// Where /api/* is proxied when the frontend is built with
// NEXT_PUBLIC_API_BASE_URL=/api, so the web app and API share one origin --
// e.g. behind a single ngrok tunnel. Read at build time.
const apiProxyTarget = (process.env.STUDIO_API_PROXY_TARGET || 'http://127.0.0.1:8189').replace(/\/+$/, '');

const nextConfig: NextConfig = {
  // Point tracing at the monorepo root (apps/fastvideo_studio -> apps -> repo
  // root) so Next stops warning about multiple lockfiles in the workspace.
  outputFileTracingRoot: path.join(configDir, '..', '..'),
  // `next dev` blocks its dev-only assets/HMR for pages opened from a host
  // other than localhost; allow the ngrok tunnel's domains. No effect in prod.
  allowedDevOrigins: ['*.ngrok-free.app', '*.ngrok.app', '*.ngrok.io'],
  async rewrites() {
    return [{ source: '/api/:path*', destination: `${apiProxyTarget}/api/:path*` }];
  },
};

export default nextConfig;
