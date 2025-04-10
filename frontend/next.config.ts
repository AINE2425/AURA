import type { NextConfig } from "next";
import UnoCSS from '@unocss/webpack';

const nextConfig: NextConfig = {
    reactStrictMode: true,
    webpack: (config) => {
        config.plugins.push(UnoCSS());
        config.cache = false; // have to be false for hmr
        return config;
    },
};

export default nextConfig;
