import { cp, rm } from "node:fs/promises";

const targets = ["index.html", "assets", "images", ".nojekyll"];

for (const target of targets) {
  await rm(new URL(`../${target}`, import.meta.url), {
    force: true,
    recursive: true,
  });
}

await cp(new URL("../dist/index.html", import.meta.url), new URL("../index.html", import.meta.url));
await cp(new URL("../dist/.nojekyll", import.meta.url), new URL("../.nojekyll", import.meta.url));
await cp(new URL("../dist/assets", import.meta.url), new URL("../assets", import.meta.url), {
  recursive: true,
});
await cp(new URL("../dist/images", import.meta.url), new URL("../images", import.meta.url), {
  recursive: true,
});