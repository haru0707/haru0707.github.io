import { rm } from "node:fs/promises";

const targets = ["index.html", "assets", "images", ".nojekyll"];

for (const target of targets) {
  await rm(new URL(`../${target}`, import.meta.url), {
    force: true,
    recursive: true,
  });
}