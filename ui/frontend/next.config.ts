import type { NextConfig } from "next";

// Check for NEXT_PUBLIC_API_BASE_URL environment variable
if (!process.env.NEXT_PUBLIC_API_BASE_URL) {
  throw new Error(
    "Please set NEXT_PUBLIC_API_BASE_URL in your .env.local file or as an environment variable. " +
    "Example: NEXT_PUBLIC_API_BASE_URL=http://localhost:8189/api"
  );
}

const nextConfig: NextConfig = {
  /* config options here */
  reactCompiler: true,
};

export default nextConfig;
