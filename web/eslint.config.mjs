// eslint.config.mjs — ESLint 9 flat config for the Crucible web Next.js app
//
// ESLint 9 flat config requires:
//   1. Global ignores as a top-level { ignores: [...] } object (NOT nested).
//   2. File extension filtering via the `files` glob (NOT --ext flag in CLI).
//   3. @typescript-eslint v8 for TypeScript support.

import tsPlugin from "@typescript-eslint/eslint-plugin";
import tsParser from "@typescript-eslint/parser";

export default [
  // ── 1. Global ignores (must be top-level, no other keys) ──────────────────
  {
    ignores: [
      "node_modules/**",
      ".next/**",
      "public/wasm/**",
      "out/**",
      "*.config.js",
      "*.config.mjs",
      "*.config.ts",
    ],
  },

  // ── 2. TypeScript source rules ─────────────────────────────────────────────
  {
    // `files` glob replaces the legacy --ext CLI flag in ESLint 9
    files: ["src/**/*.{ts,tsx}"],
    languageOptions: {
      parser: tsParser,
      parserOptions: {
        ecmaVersion: "latest",
        sourceType: "module",
        ecmaFeatures: { jsx: true },
      },
    },
    plugins: {
      "@typescript-eslint": tsPlugin,
    },
    rules: {
      // TypeScript recommended rules (no type-information needed — runs fast)
      "@typescript-eslint/no-explicit-any": "warn",
      "@typescript-eslint/no-unused-vars": ["warn", { argsIgnorePattern: "^_" }],
      "no-console": "off",
      "no-undef": "off", // TypeScript handles undefined-variable detection
    },
  },
];
