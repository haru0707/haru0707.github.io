import { type CSSProperties, useEffect, useRef, useState } from "react";
import styles from "./AttentionHero.module.css";
import { tinyLm } from "./tinyLmData";

const TOKENS: readonly string[] = tinyLm.tokens;
const TOKEN_IDS: readonly number[] = tinyLm.ids;
const N = TOKENS.length;
const VECTORS: readonly (readonly number[])[] = tinyLm.embedding;
const DIM = VECTORS[0].length;
const VECTOR_SCALE: number = tinyLm.embeddingScale;
const NEXT: readonly { token: string; p: number }[] = tinyLm.next;

// Colour and reading for each exported head. The notes describe what these
// particular heads do on this sentence (see the weights in tinyLmData.ts).
const HEAD_STYLE = [
  { color: "#f2c46d", note: "「と」「を」などの助詞に強く注目するヘッドです。" },
  { color: "#6eb6ff", note: "ひとつ前のトークンを見て、語順を伝えるヘッドです。" },
  { color: "#74dfba", note: "「体験」「静かに」など、意味を担う語に注目するヘッドです。" },
] as const;

const HEADS = tinyLm.heads.map((head, index) => ({
  ...HEAD_STYLE[index],
  label: `L${head.layer} H${head.head}`,
  weights: head.weights as readonly (readonly number[])[],
}));

// One loop, in seconds. Each scene hands its end state to the next one, and
// the last frame (the sentence at the centre) is also the first frame.
const SPLIT_AT = 1.0;
const EMBED_AT = 2.2;
const KEYS_AT = 4.6;
const HEADS_AT = 5.2;
const QUERY_LEN = 0.3;
const SWEEP_LEN = QUERY_LEN * N;
const HEAD_LEN = SWEEP_LEN + 0.45;
const MIX_AT = HEADS_AT + HEAD_LEN * HEADS.length;
const OUT_AT = MIX_AT + 1.9;
const LOOP = OUT_AT + 4.0;
const BOUNDS = [0, EMBED_AT, HEADS_AT, MIX_AT, OUT_AT, LOOP] as const;
const STILL_FRAME = HEADS_AT + HEAD_LEN * 2 + QUERY_LEN * 5.6;

const STAGES = [
  {
    title: "TOKENIZE",
    formula: `SentencePiece → ${N} tokens`,
    note: `文を ${N} 個のトークンに分けます。先頭の「▁」は空白を表す記号です。`,
  },
  {
    title: "EMBED",
    formula: "xᵢ = E[tokenᵢ] ∈ ℝ²⁵⁶",
    note: `各トークンを 256 次元のベクトルに変換します（先頭 ${DIM} 次元を表示）。`,
  },
  {
    title: "SELF-ATTENTION",
    formula: "A = softmax(QKᵀ / √dₖ)",
    note: "",
  },
  {
    title: "RESIDUAL STREAM",
    formula: "x ← x + Attn(x) + MLP(x)",
    note: "各ヘッドの結果を残差として足し合わせ、4 層かけて文脈をまとめます。",
  },
  {
    title: "NEXT TOKEN",
    formula: "p = softmax(W · RMSNorm(x))",
    note: "最後のトークンの状態から、続きの 1 トークンを予測します。",
  },
] as const;

type Rgb = [number, number, number];

const WHITE: Rgb = [255, 255, 255];
const MINT: Rgb = [116, 223, 186];
const SANS = '"DM Sans", "Noto Sans JP", sans-serif';
const MONO = '"JetBrains Mono", ui-monospace, monospace';

function hexToRgb(hex: string): Rgb {
  const value = parseInt(hex.slice(1), 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
}

const HEAD_RGB = HEADS.map((head) => hexToRgb(head.color));
const WEIGHTS = HEADS.map((head) => head.weights);
const NEXT_MAX = Math.max(...NEXT.map((item) => item.p));

const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));
const seg = (t: number, from: number, to: number) => clamp((t - from) / (to - from), 0, 1);
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const easeIn = (x: number) => x * x * x;
const easeOut = (x: number) => 1 - Math.pow(1 - x, 3);
const easeInOut = (x: number) => (x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2);
const easeBack = (x: number) => 1 + 2.4 * Math.pow(x - 1, 3) + 1.4 * Math.pow(x - 1, 2);
const rgba = (rgb: Rgb, alpha: number) =>
  `rgba(${rgb[0] | 0}, ${rgb[1] | 0}, ${rgb[2] | 0}, ${clamp(alpha, 0, 1).toFixed(3)})`;

function roundedRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function cubicPoint(x0: number, y0: number, x1: number, y1: number, t: number) {
  const mx = (x0 + x1) / 2;
  const u = 1 - t;
  const b = 3 * u * u * t;
  const c = 3 * u * t * t;
  return [
    u * u * u * x0 + (b + c) * mx + t * t * t * x1,
    (u * u * u + b) * y0 + (c + t * t * t) * y1,
  ] as const;
}

// A soft dot, drawn once and reused: far cheaper than canvas shadows.
function makeGlow(rgb: Rgb) {
  const sprite = document.createElement("canvas");
  sprite.width = sprite.height = 64;
  const g = sprite.getContext("2d");
  if (g) {
    const gradient = g.createRadialGradient(32, 32, 0, 32, 32, 32);
    gradient.addColorStop(0, rgba(rgb, 1));
    gradient.addColorStop(0.22, rgba(rgb, 0.42));
    gradient.addColorStop(1, rgba(rgb, 0));
    g.fillStyle = gradient;
    g.fillRect(0, 0, 64, 64);
  }
  return sprite;
}

function stageAt(t: number) {
  for (let index = BOUNDS.length - 2; index > 0; index--) {
    if (t >= BOUNDS[index]) return index;
  }
  return 0;
}

type Props = {
  active: boolean;
};

export function AttentionHero({ active }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fillRefs = useRef<(HTMLSpanElement | null)[]>([]);
  const [ui, setUi] = useState({ stage: 0, head: 0 });

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;

    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const headGlow = HEAD_RGB.map(makeGlow);
    const whiteGlow = makeGlow(WHITE);

    let width = 0;
    let height = 0;
    let raf = 0;
    let last = 0;
    let time = reduced ? STILL_FRAME + LOOP : 0;
    let frozen = false;
    let visible = true;
    let shownStage = -1;
    let shownHead = -1;

    // Values that chase a target, so quick scene changes still read as soft.
    const weights = new Array<number>(N).fill(0);
    const lit = new Array<number>(N).fill(0);
    const tint: Rgb = [...WHITE];
    let mixBar = 0;
    let reading = 0;

    const dot = (sprite: HTMLCanvasElement, x: number, y: number, radius: number, alpha: number) => {
      if (alpha <= 0.01) return;
      ctx.globalAlpha = Math.min(1, alpha);
      ctx.drawImage(sprite, x - radius, y - radius, radius * 2, radius * 2);
      ctx.globalAlpha = 1;
    };

    const render = (dt: number) => {
      const t = time % LOOP;
      const first = time < LOOP;
      const stage = stageAt(t);
      const sweep = t - HEADS_AT;
      const head = clamp(Math.floor(sweep / HEAD_LEN), 0, HEADS.length - 1);
      const headTime = sweep - head * HEAD_LEN;
      const sweeping = stage === 2 && headTime < SWEEP_LEN;
      const query = sweeping ? Math.min(N - 1, Math.floor(headTime / QUERY_LEN)) : -1;

      if (stage !== shownStage || (stage === 2 && head !== shownHead)) {
        shownStage = stage;
        shownHead = head;
        setUi({ stage, head });
      }
      fillRefs.current.forEach((fill, index) => {
        if (fill) fill.style.transform = `scaleX(${seg(t, BOUNDS[index], BOUNDS[index + 1]).toFixed(4)})`;
      });

      const chase = reduced ? 1 : 1 - Math.exp(-dt * 11);
      const tintTarget = stage === 2 ? HEAD_RGB[head] : WHITE;
      for (let c = 0; c < 3; c++) tint[c] += (tintTarget[c] - tint[c]) * chase;
      for (let i = 0; i < N; i++) {
        weights[i] += ((query >= 0 ? WEIGHTS[head][query][i] : 0) - weights[i]) * chase;
        lit[i] += ((i === query ? 1 : 0) - lit[i]) * chase;
      }
      mixBar += ((stage === 3 ? 1 : 0) - mixBar) * chase;
      reading += ((query >= 0 ? 1 : 0) - reading) * chase;

      // ---- layout -------------------------------------------------------
      const wide = width >= 460;
      const unit = clamp(width / 760, 0.55, 1); // line and glow sizes follow the stage
      const padLeft = wide ? 22 : 10;
      const padRight = wide ? 60 : 10;
      const top = 56;
      const bottom = 28;
      const gap = (height - top - bottom) / (N - 1);
      const pillW = clamp(width * 0.17, 60, 124);
      const pillH = clamp(gap * 0.66, 20, 40);
      const smallFont = clamp(pillH * 0.44, 10.5, 16);
      const bigFont = Math.min(width * 0.066, height * 0.15, 52);
      const leftX = padLeft + pillW / 2;
      const rightX = width - padRight - pillW / 2;
      const cell = wide ? clamp(pillH * 0.24, 5, 8) : 4;
      const cellStep = cell + 2;
      const showIds = width >= 600;
      const idsX = padLeft + pillW + 10;
      const cellsX = padLeft + pillW + (wide ? 12 : 7) + (showIds ? 44 : 0);
      const x0 = cellsX + DIM * cellStep + (wide ? 10 : 4);
      const x1 = rightX - pillW / 2 - (wide ? 10 : 4);
      const midX = (x0 + x1) / 2;
      const rowY = (index: number) => top + gap * index;

      ctx.font = `700 ${bigFont}px ${SANS}`;
      // The whitespace marker is a real token, but it has no glyph in the sentence.
      // The sentence is written solid; the gaps open just before it splits.
      const gapOpen =
        t < OUT_AT ? easeInOut(seg(t, 0.3, 0.9)) : 1 - easeInOut(seg(t, OUT_AT + 1.05, OUT_AT + 1.6));
      const tokenGap = bigFont * 0.2 * gapOpen;
      const inSentence = TOKENS.map((token) => token !== "\u2581");
      const tokenWidths = TOKENS.map((token, i) => (inSentence[i] ? ctx.measureText(token).width : 0));
      const sentenceW =
        tokenWidths.reduce((a, b) => a + b, 0) + tokenGap * (inSentence.filter(Boolean).length - 1);
      const sentenceX = (width - sentenceW) / 2;
      const centerY = height * 0.44;
      const centerX: number[] = [];
      let cursor = sentenceX;
      for (let i = 0; i < N; i++) {
        centerX.push(cursor + tokenWidths[i] / 2);
        if (inSentence[i]) cursor += tokenWidths[i] + tokenGap;
      }

      ctx.clearRect(0, 0, width, height);
      ctx.textBaseline = "middle";
      ctx.lineCap = "round";

      // ---- soft light behind the finished sentence ----------------------
      const presence =
        t < OUT_AT
          ? first
            ? 0
            : 1 - seg(t, SPLIT_AT - 0.3, SPLIT_AT + 0.4)
          : easeInOut(seg(t, OUT_AT + 0.7, OUT_AT + 1.5));
      if (presence > 0.01) {
        const halo = ctx.createRadialGradient(width / 2, centerY, 0, width / 2, centerY, sentenceW * 0.75);
        halo.addColorStop(0, `rgba(255, 255, 255, ${(0.085 * presence).toFixed(3)})`);
        halo.addColorStop(1, "rgba(255, 255, 255, 0)");
        ctx.fillStyle = halo;
        ctx.fillRect(0, 0, width, height);
      }

      // ---- attention edges ----------------------------------------------
      const edgesOut = 1 - seg(t, OUT_AT, OUT_AT + 0.5);
      const mixEdges =
        easeOut(seg(t, MIX_AT, MIX_AT + 0.4)) * lerp(1, 0.22, seg(t, MIX_AT + 0.2, MIX_AT + 0.7)) * edgesOut;

      const traceEdge = (i: number, j: number, upTo: number) => {
        const ya = rowY(i);
        const yb = rowY(j);
        ctx.beginPath();
        ctx.moveTo(x0, ya);
        if (upTo >= 1) {
          ctx.bezierCurveTo(midX, ya, midX, yb, x1, yb);
          return [x1, yb] as const;
        }
        const steps = Math.max(1, Math.round(28 * upTo));
        let tip = [x0, ya] as readonly [number, number];
        for (let s = 1; s <= steps; s++) {
          tip = cubicPoint(x0, ya, x1, yb, (s / steps) * upTo);
          ctx.lineTo(tip[0], tip[1]);
        }
        return tip;
      };

      const comets = (
        sprite: HTMLCanvasElement,
        i: number,
        j: number,
        w: number,
        upTo: number,
        alpha: number,
        halo: number,
      ) => {
        const count = Math.max(1, Math.round(w * 5));
        for (let k = 0; k < count; k++) {
          const p = (time * 0.3 + k / count + j * 0.17 + i * 0.29) % 1;
          if (p > upTo) continue;
          const fade = Math.sin(Math.PI * p) * alpha;
          for (let tail = 0; tail < 5; tail++) {
            const q = p - tail * 0.016;
            if (q < 0) break;
            const [px, py] = cubicPoint(x0, rowY(i), x1, rowY(j), q);
            ctx.fillStyle = `rgba(255, 255, 255, ${(fade * (1 - tail / 5) * 0.85).toFixed(3)})`;
            ctx.beginPath();
            ctx.arc(px, py, (1.2 + w * 1.5) * (1 - tail * 0.13) * lerp(0.75, 1, unit), 0, Math.PI * 2);
            ctx.fill();
            if (tail === 0) dot(sprite, px, py, (7 + w * 9) * unit, fade * 0.55 * halo);
          }
        }
      };

      ctx.globalCompositeOperation = "lighter";
      for (let h = 0; h < HEADS.length; h++) {
        const rgb = HEAD_RGB[h];
        const local = t - (HEADS_AT + h * HEAD_LEN);
        const inWindow = local >= 0 && local < HEAD_LEN;
        const current = inWindow ? Math.min(N - 1, Math.floor(local / QUERY_LEN)) : -1;
        const settle = inWindow ? 1 - easeInOut(seg(local, SWEEP_LEN, HEAD_LEN)) : 0;

        for (let i = 0; i < N; i++) {
          const isActive = inWindow && i === current && local < SWEEP_LEN;
          const isTrace = inWindow && !isActive && i <= current;
          if (!isActive && !isTrace && mixEdges <= 0.01) continue;

          for (let j = 0; j <= i; j++) {
            const w = WEIGHTS[h][i][j];
            if (w < 0.05) continue;

            if (isActive) {
              const grow = easeOut(seg(local - current * QUERY_LEN, 0, 0.26));
              const tip = traceEdge(i, j, grow);
              ctx.strokeStyle = rgba(rgb, 0.04 + w * 0.17);
              ctx.lineWidth = (4 + w * 16) * unit;
              ctx.stroke();
              const stroke = ctx.createLinearGradient(x0, 0, x1, 0);
              stroke.addColorStop(0, rgba(rgb, 0.9));
              stroke.addColorStop(1, rgba(rgb, 0.22 + w * 0.78));
              ctx.strokeStyle = stroke;
              ctx.lineWidth = 0.8 + w * 3.2 * unit;
              ctx.stroke();
              if (grow < 1) dot(headGlow[h], tip[0], tip[1], (8 + w * 12) * unit, 0.8 * (1 - grow));
              comets(headGlow[h], i, j, w, grow, 1, 1);
            } else if (isTrace) {
              traceEdge(i, j, 1);
              ctx.strokeStyle = rgba(rgb, w * 0.46 * settle);
              ctx.lineWidth = 0.6 + w * 2.2;
              ctx.stroke();
            } else {
              traceEdge(i, j, 1);
              ctx.strokeStyle = rgba(rgb, (0.08 + w * 0.6) * mixEdges);
              ctx.lineWidth = 0.6 + w * 2.6 * unit;
              ctx.stroke();
              if (w > 0.3) comets(headGlow[h], i, j, w, 1, mixEdges, wide ? 0.35 : 0);
            }
          }
        }
      }
      ctx.globalCompositeOperation = "source-over";

      // ---- ports where edges leave and arrive ---------------------------
      const ports = seg(t, KEYS_AT + 0.2, KEYS_AT + 0.7) * edgesOut;
      if (ports > 0.01) {
        for (let i = 0; i < N; i++) {
          const arrive = Math.max(Math.min(1, weights[i] * 1.9), mixBar);
          ctx.fillStyle = rgba(tint, (0.22 + 0.78 * lit[i]) * ports);
          ctx.beginPath();
          ctx.arc(x0, rowY(i), 1.8 + lit[i] * 1.4, 0, Math.PI * 2);
          ctx.fill();
          ctx.fillStyle = rgba(tint, (0.22 + 0.78 * arrive) * ports);
          ctx.beginPath();
          ctx.arc(x1, rowY(i), 1.8 + arrive * 1.4, 0, Math.PI * 2);
          ctx.fill();
          if (stage === 2) {
            ctx.globalCompositeOperation = "lighter";
            dot(headGlow[head], x0, rowY(i), 16 * unit, lit[i] * 0.7 * ports);
            dot(headGlow[head], x1, rowY(i), (8 + arrive * 10) * unit, arrive * 0.5 * ports);
            ctx.globalCompositeOperation = "source-over";
          }
        }
      }

      // ---- embedding cells ----------------------------------------------
      const cellsOut = 1 - seg(t, OUT_AT, OUT_AT + 0.6);
      if (showIds && t >= SPLIT_AT && cellsOut > 0.01) {
        ctx.font = `500 9.5px ${MONO}`;
        ctx.textAlign = "left";
        for (let i = 0; i < N; i++) {
          const enter = easeOut(seg(t, EMBED_AT - 0.25 + i * 0.05, EMBED_AT + 0.15 + i * 0.05));
          if (enter <= 0.01) continue;
          ctx.fillStyle = rgba(
            [lerp(245, tint[0], lit[i]), lerp(245, tint[1], lit[i]), lerp(245, tint[2], lit[i])],
            (0.34 + 0.56 * lit[i]) * enter * cellsOut,
          );
          ctx.fillText(String(TOKEN_IDS[i]), idsX - (1 - enter) * 6, rowY(i) + 1);
        }
      }
      const wave = seg(t, EMBED_AT + 0.8, EMBED_AT + 1.1) * (1 - seg(t, KEYS_AT + 0.1, KEYS_AT + 0.6));
      const settleCells = 1 - 0.45 * seg(t, KEYS_AT + 0.3, KEYS_AT + 0.9);
      // On wide stages the vectors first appear as a table of real values,
      // then fold down into the small strips the attention scene uses.
      const spread = wide ? 1 - easeInOut(seg(t, KEYS_AT - 0.7, KEYS_AT)) : 0;
      const bigStep = Math.min((width - padLeft - cellsX - 6) / DIM, 86);
      const numbers = bigStep >= 44 ? seg(spread, 0.55, 0.9) : 0;
      if (numbers > 0.01 && t >= EMBED_AT) {
        const header = numbers * easeOut(seg(t, EMBED_AT + 0.1, EMBED_AT + 0.6));
        ctx.font = `500 10px ${MONO}`;
        ctx.textAlign = "center";
        ctx.fillStyle = `rgba(245, 245, 245, ${(0.42 * header).toFixed(3)})`;
        for (let k = 0; k < DIM; k++) {
          ctx.fillText(k === DIM - 1 ? `d${k} \u2026 d255` : `d${k}`, cellsX + k * bigStep + (bigStep - 6) / 2, 24);
        }
      }
      if (t >= EMBED_AT && cellsOut > 0.01) {
        for (let i = 0; i < N; i++) {
          for (let k = 0; k < DIM; k++) {
            const startAt = EMBED_AT + 0.1 + i * 0.05 + k * 0.035;
            const pop = easeBack(seg(t, startAt, startAt + 0.34));
            if (pop <= 0.01) continue;
            const ripple = wave * (0.5 + 0.5 * Math.sin(k * 0.9 + i * 0.6 - t * 6));
            const glow = lit[i];
            const color: Rgb = [
              lerp(lerp(255, MINT[0], ripple), tint[0], glow),
              lerp(lerp(255, MINT[1], ripple), tint[1], glow),
              lerp(lerp(255, MINT[2], ripple), tint[2], glow),
            ];
            const value = VECTORS[i][k];
            const strength =
              ((0.1 + 0.75 * Math.abs(value)) * settleCells + ripple * 0.3 + glow * 0.3) * cellsOut;
            const fullW = lerp(cell, bigStep - 6, spread);
            const cellW = fullW * pop;
            const cellH = lerp(cell, pillH, spread) * pop;
            const cellX = cellsX + k * lerp(cellStep, bigStep, spread) + (fullW - cellW) / 2;
            const cellY = rowY(i) - cellH / 2;
            const radius = Math.min(lerp(1.6, 7, spread), cellW / 2, cellH / 2);
            if (value >= 0) {
              roundedRect(ctx, cellX, cellY, cellW, cellH, radius);
              ctx.fillStyle = rgba(color, strength * (1 - 0.45 * spread));
              ctx.fill();
            } else if (cellW > 2) {
              roundedRect(ctx, cellX + 0.5, cellY + 0.5, cellW - 1, cellH - 1, radius);
              ctx.strokeStyle = rgba(color, strength);
              ctx.lineWidth = 1;
              ctx.stroke();
            }
            if (numbers > 0.01) {
              const raw = value * VECTOR_SCALE;
              ctx.font = `500 10px ${MONO}`;
              ctx.textAlign = "center";
              ctx.fillStyle = `rgba(255, 255, 255, ${((0.5 + 0.5 * Math.abs(value)) * numbers * Math.min(1, pop) * cellsOut).toFixed(3)})`;
              ctx.fillText(`${raw < 0 ? "\u2212" : "+"}${Math.abs(raw).toFixed(2)}`, cellX + cellW / 2, rowY(i) + 1);
            }
          }
        }
      }

      // ---- attention matrices: one per head, then merged ----------------
      const matricesOut = seg(t, MIX_AT + 1.5, MIX_AT + 1.9);
      if (t >= MIX_AT && matricesOut < 1) {
        const gapCard = wide ? 22 : 6;
        const padCard = wide ? 9 : 5;
        const room = (x1 - x0 - 24 - gapCard * 2) / 3;
        const stepM = clamp(Math.min((room - padCard * 2) / N, gap * 0.24 + 2), 4, 14);
        const cellM = stepM - (stepM > 7 ? 2 : 1);
        const side = N * stepM - (stepM - cellM);
        const card = side + padCard * 2;
        const merge = easeInOut(seg(t, MIX_AT + 0.8, MIX_AT + 1.3));
        const breathe = 1 + 0.05 * Math.sin(Math.PI * seg(t, MIX_AT + 1.25, MIX_AT + 1.6));
        const scale = breathe * lerp(1, 0.72, easeIn(matricesOut));
        const middleY = (rowY(0) + rowY(N - 1)) / 2;

        for (let h = 0; h < HEADS.length; h++) {
          const enter = easeOut(seg(t, MIX_AT + 0.12 + h * 0.12, MIX_AT + 0.55 + h * 0.12));
          const alpha = enter * (1 - matricesOut);
          if (alpha <= 0.01) continue;
          const cx = midX + (h - 1) * (card + gapCard) * (1 - merge);
          const cy = middleY + (1 - enter) * 10;
          const half = (card * scale) / 2;

          roundedRect(ctx, cx - half, cy - half, half * 2, half * 2, wide ? 10 : 6);
          ctx.fillStyle = `rgba(7, 7, 9, ${(0.92 * alpha).toFixed(3)})`;
          ctx.fill();
          ctx.lineWidth = 1;
          ctx.strokeStyle = rgba(HEAD_RGB[h], lerp(0.5, 0.16, merge) * alpha);
          ctx.stroke();

          ctx.font = `500 9px ${MONO}`;
          ctx.textAlign = "center";
          ctx.fillStyle = rgba(HEAD_RGB[h], 0.9 * alpha * (1 - merge));
          ctx.fillText(HEADS[h].label, cx, cy - half - 10);
          if (h === 1) {
            ctx.fillStyle = `rgba(255, 255, 255, ${(0.8 * alpha * merge).toFixed(3)})`;
            ctx.fillText("x + \u03a3 Attn", cx, cy - half - 10);
          }

          ctx.globalCompositeOperation = "lighter";
          for (let i = 0; i < N; i++) {
            for (let j = 0; j < N; j++) {
              const startAt = MIX_AT + 0.15 + h * 0.12 + (i + j) * 0.022;
              const pop = easeBack(seg(t, startAt, startAt + 0.3));
              if (pop <= 0.01) continue;
              const masked = j > i;
              const size = cellM * pop * scale * (masked ? 0.3 : 1);
              ctx.fillStyle = masked
                ? `rgba(255, 255, 255, ${(0.05 * alpha).toFixed(3)})`
                : rgba(HEAD_RGB[h], (0.07 + WEIGHTS[h][i][j] * 1.25) * alpha);
              ctx.fillRect(
                cx + (j - (N - 1) / 2) * stepM * scale - size / 2,
                cy + (i - (N - 1) / 2) * stepM * scale - size / 2,
                size,
                size,
              );
            }
          }
          ctx.globalCompositeOperation = "source-over";
        }
      }

      // ---- tokens --------------------------------------------------------
      const drawToken = (
        text: string,
        x: number,
        y: number,
        pill: number,
        alpha: number,
        glow: number,
        bar: number,
        sentence: number,
      ) => {
        if (alpha <= 0.01) return;
        const fontSize = lerp(bigFont, smallFont, pill);

        if (pill > 0.01) {
          const left = x - pillW / 2;
          const topY = y - pillH / 2;
          const body = alpha * pill;
          const fill = ctx.createLinearGradient(0, topY, 0, topY + pillH);
          fill.addColorStop(0, `rgba(30, 30, 34, ${(0.95 * body).toFixed(3)})`);
          fill.addColorStop(1, `rgba(11, 11, 13, ${(0.95 * body).toFixed(3)})`);
          roundedRect(ctx, left, topY, pillW, pillH, pillH / 2);
          ctx.fillStyle = fill;
          ctx.fill();
          if (glow > 0.01) {
            ctx.fillStyle = rgba(tint, 0.2 * glow * body);
            ctx.fill();
          }
          if (bar > 0.01) {
            ctx.save();
            ctx.clip();
            ctx.fillStyle = rgba(tint, (0.1 + 0.26 * bar) * body);
            ctx.fillRect(left, topY, pillW * bar, pillH);
            ctx.restore();
          }
          ctx.lineWidth = 1;
          ctx.strokeStyle = `rgba(255, 255, 255, ${(0.15 * body).toFixed(3)})`;
          ctx.stroke();
          const edge = Math.max(glow, bar * 0.9);
          if (edge > 0.01) {
            ctx.strokeStyle = rgba(tint, edge * body);
            ctx.stroke();
          }
        }

        ctx.textAlign = "center";
        const emphasis = lerp(1, 0.62 + 0.38 * Math.max(glow, bar), pill);
        const write = (weight: number, opacity: number) => {
          if (opacity <= 0.01) return;
          ctx.font = `${weight} ${fontSize}px ${SANS}`;
          ctx.fillStyle = `rgba(255, 255, 255, ${Math.min(1, opacity).toFixed(3)})`;
          ctx.fillText(text, x, y + fontSize * 0.04);
        };
        write(700, alpha * emphasis * (1 - pill) * sentence);
        write(500, alpha * emphasis * pill);
      };

      const cuts =
        gapOpen * (t < OUT_AT ? 1 - seg(t, SPLIT_AT, SPLIT_AT + 0.25) : seg(t, OUT_AT + 0.95, OUT_AT + 1.1));
      if (cuts > 0.01) {
        ctx.strokeStyle = `rgba(255, 255, 255, ${(0.3 * cuts).toFixed(3)})`;
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let i = 0; i < N - 1; i++) {
          if (!inSentence[i]) continue;
          const cutX = centerX[i] + tokenWidths[i] / 2 + tokenGap / 2;
          ctx.moveTo(cutX, centerY - bigFont * 0.3);
          ctx.lineTo(cutX, centerY + bigFont * 0.3);
        }
        ctx.stroke();
      }

      const leftOut = 1 - seg(t, OUT_AT, OUT_AT + 0.7);
      for (let i = 0; i < N; i++) {
        const startAt = SPLIT_AT + i * 0.07;
        const moveX = easeInOut(seg(t, startAt, startAt + 0.9));
        const moveY = easeInOut(seg(t, startAt + 0.1, startAt + 1));
        let alpha = leftOut;
        let y = lerp(centerY, rowY(i), moveY);
        if (first) {
          const enter = easeOut(seg(t, i * 0.05, i * 0.05 + 0.55));
          alpha *= enter;
          y += (1 - enter) * 16;
        }
        drawToken(TOKENS[i], lerp(centerX[i], leftX, moveX), y, moveX, alpha, lit[i], 0, inSentence[i] ? 1 : 0);
      }

      for (let j = 0; j < N; j++) {
        const appear = easeOut(seg(t, KEYS_AT + j * 0.06, KEYS_AT + 0.55 + j * 0.06));
        if (appear <= 0.01) continue;
        const startAt = OUT_AT + j * 0.07;
        const moveX = easeInOut(seg(t, startAt + 0.1, startAt + 1.1));
        const moveY = easeInOut(seg(t, startAt, startAt + 1));
        drawToken(
          TOKENS[j],
          lerp(rightX + (1 - appear) * 16, centerX[j], moveX),
          lerp(rowY(j), centerY, moveY),
          1 - moveX,
          appear,
          0,
          Math.max(Math.min(1, weights[j] * 1.9), mixBar * (1 - seg(t, OUT_AT, OUT_AT + 0.4))),
          inSentence[j] ? 1 : 0,
        );
      }

      // ---- weights, read out next to the keys ---------------------------
      if (wide && reading > 0.01) {
        ctx.font = `500 10px ${MONO}`;
        ctx.textAlign = "left";
        for (let j = 0; j < N; j++) {
          ctx.fillStyle = `rgba(245, 245, 245, ${((0.28 + Math.min(0.62, weights[j] * 1.5)) * reading).toFixed(3)})`;
          ctx.fillText(
            `${Math.round(weights[j] * 100).toString().padStart(2, "0")}%`,
            rightX + pillW / 2 + 12,
            rowY(j) + 1,
          );
        }
      }

      // ---- column labels -------------------------------------------------
      ctx.font = `500 10px ${MONO}`;
      const leftLabel = easeInOut(seg(t, SPLIT_AT + 0.6, SPLIT_AT + 1.2)) * leftOut;
      if (leftLabel > 0.01) {
        ctx.fillStyle = `rgba(245, 245, 245, ${(0.42 * leftLabel).toFixed(3)})`;
        ctx.textAlign = "left";
        ctx.fillText(t < HEADS_AT ? "TOKENS" : "QUERY", padLeft, 24);
      }
      const rightLabel = seg(t, KEYS_AT, KEYS_AT + 0.5) * leftOut;
      if (rightLabel > 0.01) {
        ctx.fillStyle = `rgba(245, 245, 245, ${(0.42 * rightLabel).toFixed(3)})`;
        ctx.textAlign = "right";
        ctx.fillText("KEY / VALUE", rightX + pillW / 2, 24);
      }

      // ---- underline, then the model's real next-token prediction -------
      const lineY = centerY + bigFont * 0.9;
      let lineFrom = 0;
      let lineTo = 0;
      if (t >= OUT_AT) {
        lineTo = easeInOut(seg(t, OUT_AT + 1.0, OUT_AT + 1.9));
      } else if (!first && t < SPLIT_AT) {
        lineFrom = easeInOut(seg(t, 0.1, 0.85));
        lineTo = 1;
      }
      if (lineTo - lineFrom > 0.002) {
        const from = sentenceX + sentenceW * lineFrom;
        const to = sentenceX + sentenceW * lineTo;
        ctx.strokeStyle = "rgba(255, 255, 255, 0.4)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(from, lineY);
        ctx.lineTo(to, lineY);
        ctx.stroke();
        const moving = lineTo < 1 ? to : lineFrom > 0 ? from : -1;
        if (moving >= 0) {
          ctx.globalCompositeOperation = "lighter";
          dot(whiteGlow, moving, lineY, 12, 0.8);
          ctx.globalCompositeOperation = "source-over";
        }
      }

      const predicting = t >= OUT_AT;
      const predict = predicting ? 1 : first ? 0 : 1 - seg(t, 0, 0.45);
      if (predict > 0.01) {
        const columnW = Math.min(sentenceW / NEXT.length, 112);
        const rowX = width / 2 - (columnW * NEXT.length) / 2;
        const labelY = lineY + (wide ? 28 : 20);
        const tokenY = labelY + (wide ? 32 : 24);
        const tokenFont = clamp(bigFont * 0.42, 13, 21);
        const trackW = columnW - 14;

        const labelIn = (predicting ? easeOut(seg(t, OUT_AT + 1.6, OUT_AT + 2.1)) : 1) * predict;
        ctx.font = `500 10px ${MONO}`;
        ctx.fillStyle = `rgba(245, 245, 245, ${(0.42 * labelIn).toFixed(3)})`;
        ctx.textAlign = "left";
        ctx.fillText("NEXT TOKEN", rowX + 7, labelY);
        ctx.textAlign = "right";
        ctx.fillText(`TOP ${NEXT.length}`, rowX + columnW * NEXT.length - 7, labelY);

        for (let k = 0; k < NEXT.length; k++) {
          const enter = predicting ? easeOut(seg(t, OUT_AT + 1.75 + k * 0.09, OUT_AT + 2.25 + k * 0.09)) : 1;
          const grow = predicting ? easeOut(seg(t, OUT_AT + 1.9 + k * 0.09, OUT_AT + 2.8 + k * 0.09)) : 1;
          const alpha = enter * predict;
          if (alpha <= 0.01) continue;
          const columnX = rowX + columnW * (k + 0.5);
          const y = tokenY + (1 - enter) * 8;
          const barY = y + tokenFont * 0.95;

          ctx.textAlign = "center";
          ctx.font = `${k === 0 ? 700 : 500} ${tokenFont}px ${SANS}`;
          ctx.fillStyle = `rgba(255, 255, 255, ${((k === 0 ? 1 : 0.72) * alpha).toFixed(3)})`;
          ctx.fillText(NEXT[k].token, columnX, y);

          ctx.fillStyle = `rgba(255, 255, 255, ${(0.12 * alpha).toFixed(3)})`;
          ctx.fillRect(columnX - trackW / 2, barY, trackW, 2);
          ctx.fillStyle = `rgba(255, 255, 255, ${((k === 0 ? 0.95 : 0.6) * alpha).toFixed(3)})`;
          ctx.fillRect(columnX - trackW / 2, barY, trackW * (NEXT[k].p / NEXT_MAX) * grow, 2);

          ctx.font = `500 10px ${MONO}`;
          ctx.fillStyle = `rgba(245, 245, 245, ${(0.58 * alpha).toFixed(3)})`;
          ctx.fillText(`${(NEXT[k].p * 100 * grow).toFixed(1)}%`, columnX, barY + 14);
        }

        // the most likely token, written after the sentence as a quiet suggestion
        const ghost = (predicting ? easeOut(seg(t, OUT_AT + 2.5, OUT_AT + 3.1)) : 1) * predict;
        if (ghost > 0.01) {
          const ghostX = sentenceX + sentenceW + bigFont * 0.1;
          ctx.font = `700 ${bigFont}px ${SANS}`;
          ctx.textAlign = "left";
          ctx.fillStyle = `rgba(255, 255, 255, ${(0.3 * ghost).toFixed(3)})`;
          ctx.fillText(NEXT[0].token, ghostX, centerY + bigFont * 0.04);
          const blink = Math.sin(time * 5.5) > 0 ? 0.75 : 0.12;
          ctx.fillStyle = `rgba(255, 255, 255, ${(blink * ghost).toFixed(3)})`;
          ctx.fillRect(ghostX - bigFont * 0.05, centerY - bigFont * 0.46, 1.5, bigFont * 0.92);
        }
      }
    };

    if (import.meta.env.DEV) {
      (window as unknown as { __heroSeek?: (seconds: number) => void }).__heroSeek = (seconds) => {
        frozen = true;
        time = LOOP + seconds;
        for (let i = 0; i < 40; i++) render(0.05);
      };
    }

    const frame = (now: number) => {
      raf = 0;
      const dt = Math.min(0.05, (now - last) / 1000);
      if (!frozen) time += dt;
      render(dt);
      schedule();
    };

    const schedule = () => {
      if (raf || reduced || !active || !visible || document.hidden) return;
      last = performance.now();
      raf = requestAnimationFrame(frame);
    };

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      width = rect.width;
      height = rect.height;
      canvas.width = Math.max(1, Math.round(width * dpr));
      canvas.height = Math.max(1, Math.round(height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      render(0);
    };

    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(canvas);
    const intersectionObserver = new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting;
      schedule();
    });
    intersectionObserver.observe(canvas);
    document.addEventListener("visibilitychange", schedule);

    // Canvas text does not trigger webfont loading by itself.
    const sentence = tinyLm.text + NEXT[0].token;
    Promise.all([
      document.fonts?.load(`700 32px "Noto Sans JP"`, sentence),
      document.fonts?.load(`500 16px "Noto Sans JP"`, sentence + NEXT.map((item) => item.token).join("")),
      document.fonts?.ready,
    ])
      .then(() => render(0))
      .catch(() => undefined);

    resize();
    schedule();

    return () => {
      if (raf) cancelAnimationFrame(raf);
      resizeObserver.disconnect();
      intersectionObserver.disconnect();
      document.removeEventListener("visibilitychange", schedule);
    };
  }, [active]);

  const stage = STAGES[ui.stage];
  const isAttention = ui.stage === 2;
  const accent = isAttention ? HEADS[ui.head].color : "#ffffff";

  return (
    <div className={styles.root} style={{ "--head": accent } as CSSProperties}>
      <div className={styles.stage}>
        <canvas
          ref={canvasRef}
          className={styles.canvas}
          role="img"
          aria-label="言語モデル tiny-lm が「AIと体験を静かにつなぐ」という文をトークンに分け、Self-Attention を経て次のトークンを予測するまでを、実際の出力値で描いたアニメーション"
        />
      </div>
      <div className={styles.caption} aria-hidden>
        <div className={styles.steps}>
          {STAGES.map((item, index) => (
            <span key={item.title} className={styles.step}>
              <span
                ref={(element) => {
                  fillRefs.current[index] = element;
                }}
                className={styles.stepFill}
              />
            </span>
          ))}
        </div>
        <div className={styles.meta}>
          <span className={`${styles.title} mono`}>
            <b>0{ui.stage + 1}</b> / 05 · {stage.title}
            {isAttention ? ` · ${HEADS[ui.head].label}` : ""}
          </span>
          <span className={`${styles.formula} mono`}>{stage.formula}</span>
        </div>
        <p key={`${ui.stage}-${ui.head}`} className={styles.note}>
          {isAttention ? HEADS[ui.head].note : stage.note}
        </p>
      </div>
      <p className={`${styles.source} mono`}>
        DATA ·{" "}
        <a href={`https://huggingface.co/${tinyLm.model}`} target="_blank" rel="noreferrer">
          {tinyLm.model}
        </a>{" "}
        · 16M params · 4 layers × 4 heads の実際の出力
      </p>
    </div>
  );
}
