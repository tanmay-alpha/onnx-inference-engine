// When Vercel builds from the repository root (`cd web && npm run build`), it
// looks for the Build Output API directory at <repo>/.vercel/output. Nitro's
// vercel preset (auto-selected on Vercel, or NITRO_PRESET=vercel locally)
// writes it to web/.vercel/output, so mirror it to the repo root.
// Other presets produce no web/.vercel/output and this script does nothing.
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const webDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const src = path.join(webDir, ".vercel/output");
const dest = path.resolve(webDir, "../.vercel/output");

if (fs.existsSync(src)) {
  fs.rmSync(dest, { recursive: true, force: true });
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  fs.cpSync(src, dest, { recursive: true });
  console.log(`Copied ${src} -> ${dest}`);
}
