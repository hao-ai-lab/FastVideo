import path from 'node:path';
import { fileURLToPath } from 'node:url';
import type { NextConfig } from 'next';

const configDir = path.dirname(fileURLToPath(import.meta.url));

const nextConfig: NextConfig = {
  // Point tracing at the monorepo root (apps/fastvideo_studio -> apps -> repo
  // root) so Next stops warning about multiple lockfiles in the workspace.
  outputFileTracingRoot: path.join(configDir, '..', '..'),
};

export default nextConfig;
