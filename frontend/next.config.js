/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Offline execution guarantee
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
};

module.exports = nextConfig;
