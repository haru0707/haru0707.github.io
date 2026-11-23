import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";
import preact from "@preact/preset-vite";

const srcDir = fileURLToPath(new URL("./src", import.meta.url));

/**
 * The full Noto Sans JP is ~680 KB across 30+ files for this page. Google
 * Fonts can serve just the glyphs we ask for, so collect every Japanese
 * character used in src/ and request only those (one file, ~110 KB).
 */
function japaneseGlyphSubset(): Plugin {
  return {
    name: "japanese-glyph-subset",
    transformIndexHtml(html) {
      const glyphs = new Set<string>();
      for (const file of readdirSync(srcDir)) {
        if (!/\.(tsx?|html)$/.test(file)) continue;
        for (const char of readFileSync(join(srcDir, file), "utf8")) {
          const code = char.codePointAt(0) ?? 0;
          const isJapanese =
            (code >= 0x3000 && code <= 0x30ff) || // punctuation, hiragana, katakana
            (code >= 0x4e00 && code <= 0x9fff) || // kanji
            (code >= 0xff00 && code <= 0xffef); // full-width forms
          if (isJapanese) glyphs.add(char);
        }
      }
      return html.replace("__JP_GLYPHS__", encodeURIComponent([...glyphs].sort().join("")));
    },
  };
}

export default defineConfig({
  root: "src",
  publicDir: "../public",
  base: "/",
  plugins: [preact(), japaneseGlyphSubset()],
  build: {
    outDir: "../dist",
    emptyOutDir: true,
  },
});
