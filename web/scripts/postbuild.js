import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Copy .output/public -> repo root public/
const src = path.resolve(__dirname, "../.output/public");
const dest = path.resolve(__dirname, "../../public");

if (fs.existsSync(src)) {
  fs.mkdirSync(dest, { recursive: true });
  fs.cpSync(src, dest, { recursive: true });
  console.log(`Successfully copied ${src} -> ${dest}`);
}

// Copy .vercel/output -> repo root .vercel/output for Vercel deployment
const vercelSrc = path.resolve(__dirname, "../.vercel/output");
const vercelDest = path.resolve(__dirname, "../../.vercel/output");

if (fs.existsSync(vercelSrc)) {
  fs.mkdirSync(vercelDest, { recursive: true });
  fs.cpSync(vercelSrc, vercelDest, { recursive: true });
  console.log(`Successfully copied ${vercelSrc} -> ${vercelDest}`);
}
