import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Static site, no SSR -- deployable as-is to Vercel/Netlify/GitHub Pages
// via `npm run build` -> dist/. Base path left as the default "/"; if this
// ships under a GitHub Pages project path (e.g. username.github.io/repo/),
// set `base: "/repo/"` here before building.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "dist",
  },
});
