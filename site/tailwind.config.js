/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        // Accent used for CTAs, stat cards, chart lines, badges -- a single
        // consistent brand color rather than picking a new one per section.
        brand: {
          50: "#eefdf5",
          100: "#d6fae6",
          200: "#aef2cf",
          300: "#76e4b0",
          400: "#3ecd8e",
          500: "#1ab173",
          600: "#0f8f5c",
          700: "#0f724c",
          800: "#115a3f",
          900: "#0f4a35",
        },
        danger: {
          400: "#f87171",
          500: "#ef4444",
          600: "#dc2626",
        },
        warn: {
          300: "#fcd34d",
          400: "#fbbf24",
          500: "#f59e0b",
        },
      },
      fontFamily: {
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "sans-serif",
        ],
        mono: [
          "JetBrains Mono",
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "monospace",
        ],
      },
    },
  },
  plugins: [],
};
