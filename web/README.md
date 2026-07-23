# Crucible Web Frontend

Modern, high-performance web frontend for **Crucible** — built with **React 19**, **Vite 8**, **TanStack Start**, **TypeScript**, and **Tailwind CSS v4**.

---

## Features

- 🧪 **Interactive WASM Playground**: Run in-browser model inference console using Crucible's Rust WebAssembly engine.
- 🛡️ **Privacy-First Fraud Detector**: Live demo processing transaction records in-browser with 0ms network latency.
- 📊 **Engine Benchmark Charts**: Interactive metric visualization comparing Crucible against ONNX Runtime and PyTorch.
- 📖 **ONNX Operator Docs**: Reference guide detailing supported ONNX operator specifications and backend implementations.

---

## Development

```bash
# Install dependencies
npm install

# Start Vite development server
npm run dev

# Run TypeScript type check
npm run type-check

# Run ESLint validation
npm run lint

# Format code with Prettier
npm run format

# Production build
npm run build
```

---

## Deployment

The app compiles using Nitro with `NITRO_PRESET=vercel` into `.vercel/output`, copied to root postbuild via `scripts/postbuild.js` for native serverless SSR on Vercel.

### Environment Variables for Vercel Dashboard

When deploying to Vercel, set the following environment variable in your project settings:

- `VITE_API_URL`: Backend API URL (e.g. `https://crucible-inference-engine.onrender.com` or Render service URL).

