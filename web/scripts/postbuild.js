import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const src = path.resolve(__dirname, "../.output/public");
const dest = path.resolve(__dirname, "../../public");

if (fs.existsSync(src)) {
  fs.mkdirSync(dest, { recursive: true });
  fs.cpSync(src, dest, { recursive: true });
  console.log(`Successfully copied ${src} -> ${dest}`);
} else {
  console.warn(`Source directory does not exist: ${src}`);
}
