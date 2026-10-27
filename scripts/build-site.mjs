// Copy only browser assets. The Python service and private data stay on the server.
import {cp, mkdir, rm} from "node:fs/promises";
import {fileURLToPath} from "node:url";

const output = fileURLToPath(new URL("../site/", import.meta.url));
await rm(output, {recursive: true, force: true});
await mkdir(new URL("../site/static/fonts/", import.meta.url), {recursive: true});
await cp(new URL("../handwriting/static/index.html", import.meta.url), new URL("../site/index.html", import.meta.url));
const files = [
  "studio.css", "studio.js", "theme.js",
  "fonts/lord-handwriting.woff2", "fonts/noto-sans-latin-wght-normal.woff2", "fonts/NotoSans-OFL.txt",
];
for (const filename of files) {
  await cp(new URL(`../handwriting/static/${filename}`, import.meta.url), new URL(`../site/static/${filename}`, import.meta.url));
}
