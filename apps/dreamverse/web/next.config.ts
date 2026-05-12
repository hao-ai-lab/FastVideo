import path from 'node:path';
import { fileURLToPath } from 'node:url';
import type { NextConfig } from 'next';

const backendHost = process.env.BACKEND_HOST || '127.0.0.1';
const backendPort = Number(process.env.BACKEND_PORT) || 8009;
const backendUrl = `http://${backendHost}:${backendPort}`;
const configDir = path.dirname(fileURLToPath(import.meta.url));

const nextConfig: NextConfig = {
  outputFileTracingRoot: path.join(configDir, '..', '..', '..'),
  async rewrites() {
    return [
      { 
        source: '/ws', 
        destination: `${backendUrl}/ws` 
      },
      { 
        source: '/healthz', 
        destination: `${backendUrl}/healthz` 
      },
      { 
        source: '/readyz', 
        destination: `${backendUrl}/readyz` 
      },
      { 
        source: '/models', 
        destination: `${backendUrl}/models` 
      },
      { 
        source: '/status', 
        destination: `${backendUrl}/status` 
      },
      { 
        source: '/router/:path*', 
        destination: `${backendUrl}/router/:path*` 
      },
      {
        source: '/prompt-system-config',
        destination: `${backendUrl}/prompt-system-config`,
      },
      {
        source: '/curated-presets',
        destination: `${backendUrl}/curated-presets`,
      },
      {
        source: '/curated-presets/:path*',
        destination: `${backendUrl}/curated-presets/:path*`,
      },
    ];
  },
  webpack: (config) => {
    config.module.rules.push({
      test: /\.jsonl$/,
      type: 'asset/source',
    });
    return config;
  },
};

export default nextConfig;
