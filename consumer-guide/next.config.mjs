/** @type {import('next').NextConfig} */
const basePath = process.env.NEXT_BASE_PATH || "/config-generator"

const nextConfig = {
  output: "export",
  trailingSlash: true,
  basePath,
  assetPrefix: basePath,
  typescript: {
    ignoreBuildErrors: false,
  },
  images: {
    unoptimized: true,
  },
}

export default nextConfig
