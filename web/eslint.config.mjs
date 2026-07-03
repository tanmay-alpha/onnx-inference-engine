// eslint.config.mjs — ESLint 9 flat config for the Crucible web Next.js app
import tsPlugin from "@typescript-eslint/eslint-plugin";
import tsParser from "@typescript-eslint/parser";

export default [
  {
    // Apply to all TypeScript / TSX source files
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
      // TypeScript recommended rules (subset that don't need type information)
      "@typescript-eslint/no-explicit-any": "warn",
      "@typescript-eslint/no-unused-vars": ["warn", { argsIgnorePattern: "^_" }],
      "no-console": "off",
      "no-undef": "off", // TypeScript handles this
    },
  },
  {
    // Ignore generated / build output directories
    ignores: [
      "node_modules/**",
      ".next/**",
      "public/wasm/**",
      "out/**",
    ],
  },
];
