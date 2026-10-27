// Copy only browser assets. The Python service and private data stay on the server.
import {cp, mkdir, rm} from "node:fs/promises";
import {fileURLToPath} from "node:url";

const output = fileURLToPath(new URL("../site/", import.meta.url));
await rm(output, {recursive: true, force: true});
await mkdir(new URL("../site/static/", import.meta.url), {recursive: true});
await cp(new URL("../handwriting/static/index.html", import.meta.url), new URL("../site/index.html", import.meta.url));
for (const filename of ["daisyui.css", "studio.css", "studio.js"]) {
  await cp(new URL(`../handwriting/static/${filename}`, import.meta.url), new URL(`../site/static/${filename}`, import.meta.url));
}
