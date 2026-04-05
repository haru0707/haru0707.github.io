import { useEffect, useId, useMemo, useRef, useState } from "react";
import styles from "./TransformerHeroAnimation.module.css";

const TOKENS = ["AI", "と", "体験", "を", "静かに", "つなぐ"] as const;
const HEAD_COLORS = ["#f2c46d", "#6eb6ff", "#74dfba"] as const;
const Q_COLOR = "#f2c46d";
const K_COLOR = "#6eb6ff";
const V_COLOR = "#74dfba";
const OUTPUT_COLOR = "#f4f6fb";

const LOOP_MS = 15000;
const MODEL_DIM = 8;
const HEADS = 3;
const HEAD_DIM = 4;
const FFN_DIM = 12;
const ZERO_POINT_3D: Point3D = { x: 0, y: 0, z: 0 };

type Matrix = number[][];
type Tensor3 = number[][][];
type Point = { x: number; y: number };
type Point3D = { x: number; y: number; z: number };
type ProjectedPoint = { x: number; y: number; depth: number };
type Angles = { yaw: number; pitch: number; roll: number };

type PipelineData = {
  inputs: Matrix;
  q: Tensor3;
  k: Tensor3;
  v: Tensor3;
  attention: Tensor3;
  headOutputs: Tensor3;
  concat: Matrix;
  final: Matrix;
};

type Stage = {
  id: string;
  step: string;
  title: string;
  formula: string;
  summary: string;
  start: number;
  end: number;
};

type ProjectedSpace = {
  cube: ProjectedPoint[];
  plane?: ProjectedPoint[];
  points: ProjectedPoint[];
  center: ProjectedPoint;
  focus: ProjectedPoint;
};

const STAGES: Stage[] = [
  {
    id: "embed",
    step: "01 / 05",
    title: "INPUT + POSITIONAL CONTEXT",
    formula: "x_i = e_i + p_i",
    summary:
      "AI と体験をつなぐ入力に、順序と距離の手がかりを重ねて文脈の土台を作ります。",
    start: 0.0,
    end: 0.2,
  },
  {
    id: "qkv",
    step: "02 / 05",
    title: "LINEAR PROJECTIONS FOR CONTEXT",
    formula: "Q = XW_Q,  K = XW_K,  V = XW_V",
    summary:
      "一つの入力を複数の見方へ写し、意味と関係を扱いやすい形へ整えます。",
    start: 0.2,
    end: 0.4,
  },
  {
    id: "attention",
    step: "03 / 05",
    title: "ATTENTION FOR RELEVANCE",
    formula: "A = softmax(QK^T / sqrt(d_k))",
    summary:
      "今注目している語がどこを見るべきかを比較し、必要な関係だけを浮かび上がらせます。",
    start: 0.4,
    end: 0.6,
  },
  {
    id: "heads",
    step: "04 / 05",
    title: "MULTI-HEAD CONTEXT MIXING",
    formula: "H_h = A_h V_h,  concat = [H_1; H_2; H_3]",
    summary:
      "異なる観点の情報を束ねて、体験へ返せる一つの文脈表現へまとめます。",
    start: 0.6,
    end: 0.8,
  },
  {
    id: "output",
    step: "05 / 05",
    title: "OUTPUT TUNED FOR EXPERIENCE",
    formula: "y = LN(x + MHA(x));  out = LN(y + FFN(y))",
    summary:
      "最後に residual と FFN で情報を整え、次の層や UI につながる出力へ磨きます。",
    start: 0.8,
    end: 1.0,
  },
];

const SPACE_CUBE_VERTICES: Point3D[] = [
  { x: -1, y: -1, z: -1 },
  { x: 1, y: -1, z: -1 },
  { x: 1, y: 1, z: -1 },
  { x: -1, y: 1, z: -1 },
  { x: -1, y: -1, z: 1 },
  { x: 1, y: -1, z: 1 },
  { x: 1, y: 1, z: 1 },
  { x: -1, y: 1, z: 1 },
];

const SPACE_CUBE_EDGES = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 0],
  [4, 5],
  [5, 6],
  [6, 7],
  [7, 4],
  [0, 4],
  [1, 5],
  [2, 6],
  [3, 7],
] as const;

const SPACE_PLANES = [
  {
    color: Q_COLOR,
    points: [
      { x: -1, y: -1, z: 0.48 },
      { x: 1, y: -1, z: 0.48 },
      { x: 1, y: 1, z: 0.48 },
      { x: -1, y: 1, z: 0.48 },
    ],
  },
  {
    color: K_COLOR,
    points: [
      { x: -0.44, y: -1, z: -1 },
      { x: -0.44, y: -1, z: 1 },
      { x: -0.44, y: 1, z: 1 },
      { x: -0.44, y: 1, z: -1 },
    ],
  },
  {
    color: V_COLOR,
    points: [
      { x: -1, y: 0.34, z: -1 },
      { x: 1, y: 0.34, z: -1 },
      { x: 1, y: 0.34, z: 1 },
      { x: -1, y: 0.34, z: 1 },
    ],
  },
] as const;

function clamp01(value: number) {
  return Math.max(0, Math.min(1, value));
}

function easeInOutCubic(value: number) {
  return value < 0.5
    ? 4 * value * value * value
    : 1 - Math.pow(-2 * value + 2, 3) / 2;
}

function easeOutCubic(value: number) {
  return 1 - Math.pow(1 - value, 3);
}

function stageOpacity(progress: number, start: number, end: number, fade = 0.04) {
  const fadeIn = easeInOutCubic(clamp01((progress - start) / fade));
  const fadeOut = easeInOutCubic(clamp01((end - progress) / fade));
  return fadeIn * fadeOut;
}

function stageProgress(progress: number, start: number, end: number) {
  return clamp01((progress - start) / (end - start));
}

function addVec(a: number[], b: number[]) {
  return a.map((value, index) => value + b[index]);
}

function dot(a: number[], b: number[]) {
  return a.reduce((sum, value, index) => sum + value * b[index], 0);
}

function matVec(matrix: Matrix, vector: number[]) {
  return matrix.map((row) => dot(row, vector));
}

function concatVectors(vectors: number[][]) {
  return vectors.flat();
}

function softmax(values: number[]) {
  const max = Math.max(...values);
  const exps = values.map((value) => Math.exp(value - max));
  const total = exps.reduce((sum, value) => sum + value, 0);
  return exps.map((value) => value / total);
}

function gelu(value: number) {
  return (
    0.5 *
    value *
    (1 +
      Math.tanh(
        Math.sqrt(2 / Math.PI) * (value + 0.044715 * Math.pow(value, 3)),
      ))
  );
}

function layerNorm(vector: number[]) {
  const mean = vector.reduce((sum, value) => sum + value, 0) / vector.length;
  const variance =
    vector.reduce((sum, value) => sum + Math.pow(value - mean, 2), 0) /
    vector.length;
  const denom = Math.sqrt(variance + 1e-5);
  return vector.map((value) => (value - mean) / denom);
}

function createMatrix(
  rows: number,
  cols: number,
  seedA: number,
  seedB: number,
  scale: number,
) {
  return Array.from({ length: rows }, (_, rowIndex) =>
    Array.from({ length: cols }, (_, colIndex) => {
      const x = rowIndex + 1;
      const y = colIndex + 1;
      const sinPart = Math.sin(seedA * x * 1.27 + seedB * y * 0.81);
      const cosPart = Math.cos(seedB * x * 0.61 - seedA * y * 1.12);
      return (sinPart * 0.72 + cosPart * 0.56) * scale;
    }),
  );
}

function positionalEncoding(position: number, dimension: number) {
  return Array.from({ length: dimension }, (_, dimIndex) => {
    const scale = Math.pow(10000, (2 * Math.floor(dimIndex / 2)) / dimension);
    const angle = position / scale;
    return dimIndex % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
  });
}

function baseEmbedding(tokenIndex: number, dimension: number) {
  return Array.from({ length: dimension }, (_, dimIndex) => {
    const x = tokenIndex + 1;
    const y = dimIndex + 1;
    return (
      Math.sin(x * 0.94 + y * 0.72) * 0.82 +
      Math.cos(x * y * 0.33) * 0.58 +
      Math.sin((x + y) * 0.41) * 0.28
    );
  });
}

function averageVectors(vectors: number[][]) {
  return vectors[0].map((_, dimIndex) => {
    return (
      vectors.reduce((sum, vector) => sum + vector[dimIndex], 0) / vectors.length
    );
  });
}

function normalizeWeights(values: number[]) {
  const length =
    Math.sqrt(values.reduce((sum, value) => sum + value * value, 0)) || 1;
  return values.map((value) => value / length);
}

function createProjectionBasis(dimension: number, seed: number) {
  const axis = (offsetA: number, offsetB: number, offsetC: number) =>
    normalizeWeights(
      Array.from({ length: dimension }, (_, index) => {
        const t = index + 1;
        return (
          Math.sin(t * (0.63 + seed * 0.11 + offsetA)) * 0.68 +
          Math.cos(t * (0.39 + seed * 0.09 + offsetB)) * 0.54 +
          Math.sin((t + seed * 1.7) * offsetC) * 0.36
        );
      }),
    );

  return [
    axis(0.11, 0.27, 0.41),
    axis(0.47, 0.18, 0.57),
    axis(0.83, 0.36, 0.29),
  ] as const;
}

function vectorToPoint3D(vector: number[], seed: number): Point3D {
  const basis = createProjectionBasis(vector.length, seed);
  return {
    x: dot(vector, basis[0]),
    y: dot(vector, basis[1]),
    z: dot(vector, basis[2]),
  };
}

function normalizePointCloud(points: Point3D[]) {
  const extent =
    Math.max(
      ...points.flatMap((point) => [
        Math.abs(point.x),
        Math.abs(point.y),
        Math.abs(point.z),
      ]),
    ) || 1;

  return points.map((point) => ({
    x: point.x / extent,
    y: point.y / extent,
    z: point.z / extent,
  }));
}

function rotatePoint3D(
  point: Point3D,
  yaw: number,
  pitch: number,
  roll: number,
): Point3D {
  const cosYaw = Math.cos(yaw);
  const sinYaw = Math.sin(yaw);
  const yawX = point.x * cosYaw + point.z * sinYaw;
  const yawZ = -point.x * sinYaw + point.z * cosYaw;

  const cosPitch = Math.cos(pitch);
  const sinPitch = Math.sin(pitch);
  const pitchY = point.y * cosPitch - yawZ * sinPitch;
  const pitchZ = point.y * sinPitch + yawZ * cosPitch;

  const cosRoll = Math.cos(roll);
  const sinRoll = Math.sin(roll);

  return {
    x: yawX * cosRoll - pitchY * sinRoll,
    y: yawX * sinRoll + pitchY * cosRoll,
    z: pitchZ,
  };
}

function projectPoint3D(point: Point3D, origin: Point, scale: number): ProjectedPoint {
  const depth = 0.82 + (point.z + 1.1) * 0.28;
  return {
    x: origin.x + point.x * scale * depth,
    y: origin.y - point.y * scale * depth,
    depth,
  };
}

function buildProjectedSpace(
  points: Point3D[],
  origin: Point,
  scale: number,
  angles: Angles,
  focusIndex: number,
  planeIndex?: number,
): ProjectedSpace {
  const sourcePoints =
    points.length > 0
      ? points.map((point) => point ?? ZERO_POINT_3D)
      : [ZERO_POINT_3D];
  const safeFocusIndex = Number.isFinite(focusIndex)
    ? Math.min(sourcePoints.length - 1, Math.max(0, Math.trunc(focusIndex)))
    : 0;
  const planePoints =
    planeIndex === undefined ? undefined : SPACE_PLANES[planeIndex]?.points;
  const project = (point: Point3D) =>
    projectPoint3D(
      rotatePoint3D(point, angles.yaw, angles.pitch, angles.roll),
      origin,
      scale,
    );

  return {
    cube: SPACE_CUBE_VERTICES.map(project),
    plane: planePoints?.map(project),
    points: sourcePoints.map(project),
    center: project(ZERO_POINT_3D),
    focus: project(sourcePoints[safeFocusIndex]),
  };
}

function orderByDepth(points: ProjectedPoint[]) {
  return points
    .map((point, index) => ({ ...point, index }))
    .sort((a, b) => a.depth - b.depth);
}

function polylinePath(points: ProjectedPoint[]) {
  return points
    .map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`)
    .join(" ");
}

function polygonString(points: ProjectedPoint[]) {
  return points.map((point) => `${point.x},${point.y}`).join(" ");
}

function curvePath(x1: number, y1: number, x2: number, y2: number, bend = 0.38) {
  const dx = x2 - x1;
  const c1x = x1 + dx * bend;
  const c2x = x2 - dx * bend;
  return `M ${x1} ${y1} C ${c1x} ${y1}, ${c2x} ${y2}, ${x2} ${y2}`;
}

function positionalWavePath(
  x: number,
  y: number,
  width: number,
  height: number,
  phaseOffset: number,
) {
  const steps = 18;
  const points = Array.from({ length: steps + 1 }, (_, step) => {
    const t = step / steps;
    const px = x + width * t;
    const py =
      y +
      height * 0.5 +
      Math.sin(t * Math.PI * 2 + phaseOffset) * height * 0.26;
    return `${step === 0 ? "M" : "L"} ${px.toFixed(2)} ${py.toFixed(2)}`;
  });

  return points.join(" ");
}

function createPipeline(): PipelineData {
  const inputs = TOKENS.map((_, tokenIndex) => {
    const base = baseEmbedding(tokenIndex, MODEL_DIM);
    const position = positionalEncoding(tokenIndex, MODEL_DIM).map(
      (value, dimIndex) => value * (dimIndex % 2 === 0 ? 0.58 : 0.42),
    );
    return addVec(base, position);
  });

  const wq = Array.from({ length: HEADS }, (_, headIndex) =>
    createMatrix(
      HEAD_DIM,
      MODEL_DIM,
      0.72 + headIndex * 0.23,
      1.21 + headIndex * 0.19,
      0.42,
    ),
  );
  const wk = Array.from({ length: HEADS }, (_, headIndex) =>
    createMatrix(
      HEAD_DIM,
      MODEL_DIM,
      1.04 + headIndex * 0.18,
      0.93 + headIndex * 0.27,
      0.37,
    ),
  );
  const wv = Array.from({ length: HEADS }, (_, headIndex) =>
    createMatrix(
      HEAD_DIM,
      MODEL_DIM,
      0.88 + headIndex * 0.21,
      1.46 + headIndex * 0.17,
      0.45,
    ),
  );

  const q = wq.map((matrix) => inputs.map((vector) => matVec(matrix, vector)));
  const k = wk.map((matrix) => inputs.map((vector) => matVec(matrix, vector)));
  const v = wv.map((matrix) => inputs.map((vector) => matVec(matrix, vector)));

  const attention = q.map((queries, headIndex) =>
    queries.map((queryVector) => {
      const scores = k[headIndex].map(
        (keyVector) => dot(queryVector, keyVector) / Math.sqrt(HEAD_DIM),
      );
      return softmax(scores);
    }),
  );

  const headOutputs = attention.map((weightsByToken, headIndex) =>
    weightsByToken.map((weights) => {
      const acc = Array.from({ length: HEAD_DIM }, () => 0);
      weights.forEach((weight, tokenIndex) => {
        v[headIndex][tokenIndex].forEach((value, dimIndex) => {
          acc[dimIndex] += value * weight;
        });
      });
      return acc;
    }),
  );

  const concat = TOKENS.map((_, tokenIndex) =>
    concatVectors(headOutputs.map((head) => head[tokenIndex])),
  );
  const wo = createMatrix(MODEL_DIM, HEADS * HEAD_DIM, 0.77, 1.91, 0.31);
  const attnProjected = concat.map((vector) => matVec(wo, vector));
  const residual = inputs.map((vector, tokenIndex) =>
    addVec(vector, attnProjected[tokenIndex]),
  );
  const normalized = residual.map(layerNorm);
  const w1 = createMatrix(FFN_DIM, MODEL_DIM, 0.61, 1.37, 0.28);
  const w2 = createMatrix(MODEL_DIM, FFN_DIM, 1.17, 0.84, 0.25);
  const ffnExpanded = normalized.map((vector) => matVec(w1, vector).map(gelu));
  const ffnOut = ffnExpanded.map((vector) => matVec(w2, vector));
  const final = normalized.map((vector, tokenIndex) =>
    layerNorm(addVec(vector, ffnOut[tokenIndex])),
  );

  return {
    inputs,
    q,
    k,
    v,
    attention,
    headOutputs,
    concat,
    final,
  };
}

function usePrefersReducedMotion() {
  const [prefersReducedMotion, setPrefersReducedMotion] = useState(false);

  useEffect(() => {
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setPrefersReducedMotion(media.matches);
    update();

    if (typeof media.addEventListener === "function") {
      media.addEventListener("change", update);
      return () => media.removeEventListener("change", update);
    }

    const legacyMedia = media as MediaQueryList & {
      addListener?: (listener: (event: MediaQueryListEvent) => void) => void;
      removeListener?: (listener: (event: MediaQueryListEvent) => void) => void;
    };

    legacyMedia.addListener?.(update);
    return () => legacyMedia.removeListener?.(update);
  }, []);

  return prefersReducedMotion;
}

function useScrollPop(reducedMotion: boolean, active: boolean) {
  const rootRef = useRef<HTMLDivElement>(null);
  const [scrollPop, setScrollPop] = useState(reducedMotion || !active ? 1 : 0);

  useEffect(() => {
    if (reducedMotion || !active) {
      setScrollPop(1);
      return;
    }

    let raf = 0;

    const measure = () => {
      raf = 0;
      const element = rootRef.current;
      if (!element) return;

      const rect = element.getBoundingClientRect();
      const viewportHeight = window.innerHeight || 1;
      const enter = clamp01(
        (viewportHeight - rect.top) / (viewportHeight + rect.height * 0.32),
      );
      const centered = clamp01(
        1 -
          Math.abs(rect.top + rect.height * 0.5 - viewportHeight * 0.5) /
            (viewportHeight * 0.92),
      );
      const raw = clamp01(enter * 0.72 + centered * 0.44);
      setScrollPop(easeOutCubic(raw));
    };

    const onScroll = () => {
      if (!raf) raf = requestAnimationFrame(measure);
    };

    measure();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);

    return () => {
      if (raf) cancelAnimationFrame(raf);
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
    };
  }, [active, reducedMotion]);

  return { rootRef, scrollPop };
}

type SpaceClusterProps = {
  space: ProjectedSpace;
  label: string;
  color: string;
  opacity: number;
  haloRadius: number;
  pathOpacity?: number;
  labelDy?: number;
  planeColor?: string;
};

function SpaceCluster({
  space,
  label,
  color,
  opacity,
  haloRadius,
  pathOpacity = 0.26,
  labelDy = -92,
  planeColor,
}: SpaceClusterProps) {
  return (
    <g opacity={opacity}>
      <ellipse
        cx={space.center.x}
        cy={space.center.y}
        rx={haloRadius}
        ry={haloRadius * 0.72}
        fill={color}
        opacity={0.04}
      />
      {space.plane ? (
        <polygon
          points={polygonString(space.plane)}
          fill={planeColor ?? color}
          stroke={planeColor ?? color}
          strokeWidth="1"
          opacity={0.06}
        />
      ) : null}
      {SPACE_CUBE_EDGES.map(([start, end]) => (
        <line
          key={`${start}-${end}`}
          x1={space.cube[start].x}
          y1={space.cube[start].y}
          x2={space.cube[end].x}
          y2={space.cube[end].y}
          stroke="rgba(245,247,251,0.15)"
          strokeWidth="1"
        />
      ))}
      <path
        d={polylinePath(space.points)}
        fill="none"
        stroke={color}
        strokeWidth="1.15"
        strokeDasharray="4 9"
        opacity={pathOpacity}
      />
      {orderByDepth(space.points).map((point) => (
        <circle
          key={point.index}
          cx={point.x}
          cy={point.y}
          r={(point.index === 0 ? 1.6 : 1.6) * point.depth}
          fill={color}
          opacity={0.34}
        />
      ))}
      <circle
        cx={space.focus.x}
        cy={space.focus.y}
        r={12 * space.focus.depth}
        fill={color}
        opacity={0.06}
      />
      <circle
        cx={space.focus.x}
        cy={space.focus.y}
        r={3.6 * space.focus.depth}
        fill={color}
        opacity={0.94}
      />
      <text
        x={space.center.x}
        y={space.center.y + labelDy}
        textAnchor="middle"
        fontFamily="JetBrains Mono, monospace"
        fontSize="12"
        letterSpacing="0.18em"
        fill={color}
        opacity={0.78}
      >
        {label}
      </text>
    </g>
  );
}

type AttentionMatrixProps = {
  x: number;
  y: number;
  size: number;
  matrix: Matrix;
  focusIndex: number;
  color: string;
  label: string;
  opacity: number;
};

function AttentionMatrix({
  x,
  y,
  size,
  matrix,
  focusIndex,
  color,
  label,
  opacity,
}: AttentionMatrixProps) {
  const cell = size / matrix.length;

  return (
    <g opacity={opacity}>
      <ellipse
        cx={x + size * 0.5}
        cy={y + size * 0.5}
        rx={size * 0.68}
        ry={size * 0.68}
        fill={color}
        opacity={0.04}
      />
      <rect
        x={x}
        y={y}
        width={size}
        height={size}
        rx={18}
        fill="rgba(255,255,255,0.012)"
        stroke="rgba(255,255,255,0.09)"
      />
      {Array.from({ length: matrix.length + 1 }, (_, index) => (
        <g key={index}>
          <line
            x1={x + index * cell}
            y1={y}
            x2={x + index * cell}
            y2={y + size}
            stroke="rgba(255,255,255,0.05)"
            strokeWidth="0.8"
          />
          <line
            x1={x}
            y1={y + index * cell}
            x2={x + size}
            y2={y + index * cell}
            stroke="rgba(255,255,255,0.05)"
            strokeWidth="0.8"
          />
        </g>
      ))}
      <rect
        x={x}
        y={y + focusIndex * cell}
        width={size}
        height={cell}
        fill="rgba(255,255,255,0.03)"
        stroke="rgba(255,255,255,0.12)"
      />
      {matrix.map((row, rowIndex) =>
        row.map((value, colIndex) => (
          <rect
            key={`${rowIndex}-${colIndex}`}
            x={x + colIndex * cell + 4}
            y={y + rowIndex * cell + 4}
            width={cell - 8}
            height={cell - 8}
            rx={5}
            fill={color}
            opacity={0.08 + value * 0.88}
          />
        )),
      )}
      {TOKENS.map((token, index) => (
        <g key={token}>
          <text
            x={x + cell * (index + 0.5)}
            y={y - 14}
            textAnchor="middle"
            fontFamily="DM Sans, system-ui, sans-serif"
            fontSize="12"
            fill="rgba(245,247,251,0.72)"
          >
            {token}
          </text>
          <text
            x={x - 14}
            y={y + cell * (index + 0.5) + 4}
            textAnchor="end"
            fontFamily="DM Sans, system-ui, sans-serif"
            fontSize="12"
            fill={index === focusIndex ? "rgba(245,247,251,0.94)" : "rgba(245,247,251,0.58)"}
          >
            {token}
          </text>
        </g>
      ))}
      <text
        x={x + size * 0.5}
        y={y + size + 26}
        textAnchor="middle"
        fontFamily="JetBrains Mono, monospace"
        fontSize="12"
        letterSpacing="0.18em"
        fill={color}
        opacity={0.82}
      >
        {label}
      </text>
    </g>
  );
}

type TransformerHeroAnimationProps = {
  active?: boolean;
};

export function TransformerHeroAnimation({
  active = true,
}: TransformerHeroAnimationProps) {
  const pipeline = useMemo(() => createPipeline(), []);
  const reducedMotion = usePrefersReducedMotion();
  const { rootRef, scrollPop } = useScrollPop(reducedMotion, active);
  const [elapsed, setElapsed] = useState(0);
  const ids = useId().replace(/:/g, "");

  useEffect(() => {
    if (!active) {
      setElapsed(0);
      return;
    }

    if (reducedMotion) {
      setElapsed(LOOP_MS * 4.08);
      return;
    }

    let frame = 0;
    const start = performance.now();

    const tick = (now: number) => {
      setElapsed(now - start);
      frame = requestAnimationFrame(tick);
    };

    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [active, reducedMotion]);

  const progress = !active ? 0 : reducedMotion ? 0.88 : (elapsed % LOOP_MS) / LOOP_MS;
  const cycle = !active ? 0 : reducedMotion ? 3 : Math.floor(elapsed / LOOP_MS);
  const rawFocusIndex = !active ? 0 : reducedMotion ? 2 : cycle % TOKENS.length;
  const focusIndex = Number.isFinite(rawFocusIndex)
    ? Math.min(TOKENS.length - 1, Math.max(0, Math.trunc(rawFocusIndex)))
    : 0;
  const focusToken = TOKENS[focusIndex];

  const currentStage = STAGES.reduce((best, stage) => {
    const bestOpacity = stageOpacity(progress, best.start, best.end);
    const nextOpacity = stageOpacity(progress, stage.start, stage.end);
    return nextOpacity > bestOpacity ? stage : best;
  }, STAGES[0]);

  const scrollDepth = reducedMotion || !active ? 1 : scrollPop;
  const scrollLift = 1 - scrollDepth;

  const stageMap = Object.fromEntries(
    STAGES.map((stage) => [
      stage.id,
      {
        opacity: stageOpacity(progress, stage.start, stage.end),
        progress: stageProgress(progress, stage.start, stage.end),
      },
    ]),
  ) as Record<string, { opacity: number; progress: number }>;

  const vectorSpaces = useMemo(
    () => ({
      input: normalizePointCloud(
        pipeline.inputs.map((vector) => vectorToPoint3D(vector, 0.61)),
      ),
      qAll: normalizePointCloud(
        TOKENS.map((_, index) =>
          vectorToPoint3D(averageVectors(pipeline.q.map((head) => head[index])), 1.03),
        ),
      ),
      kAll: normalizePointCloud(
        TOKENS.map((_, index) =>
          vectorToPoint3D(averageVectors(pipeline.k.map((head) => head[index])), 1.27),
        ),
      ),
      vAll: normalizePointCloud(
        TOKENS.map((_, index) =>
          vectorToPoint3D(averageVectors(pipeline.v.map((head) => head[index])), 1.51),
        ),
      ),
      heads: pipeline.headOutputs.map((head, index) =>
        normalizePointCloud(
          head.map((vector) => vectorToPoint3D(vector, 1.79 + index * 0.21)),
        ),
      ),
      mix: normalizePointCloud(
        pipeline.concat.map((vector) => vectorToPoint3D(vector, 2.04)),
      ),
      output: normalizePointCloud(
        pipeline.final.map((vector) => vectorToPoint3D(vector, 2.28)),
      ),
    }),
    [pipeline],
  );

  const angles: Angles = {
    yaw: -0.78 + (reducedMotion ? 0 : Math.sin(elapsed * 0.00024) * 0.2),
    pitch: -0.32 + (reducedMotion ? 0 : Math.cos(elapsed * 0.00018) * 0.08),
    roll: 0.12 + (reducedMotion ? 0 : Math.sin(elapsed * 0.00013) * 0.06),
  };

  const embedSpace = buildProjectedSpace(
    vectorSpaces.input,
    { x: 992, y: 394 },
    124,
    angles,
    focusIndex,
    0,
  );
  const projectionInputSpace = buildProjectedSpace(
    vectorSpaces.input,
    { x: 394, y: 394 },
    82,
    angles,
    focusIndex,
    0,
  );
  const qSpace = buildProjectedSpace(
    vectorSpaces.qAll,
    { x: 930, y: 236 },
    62,
    { yaw: angles.yaw - 0.14, pitch: angles.pitch, roll: angles.roll },
    focusIndex,
    0,
  );
  const kSpace = buildProjectedSpace(
    vectorSpaces.kAll,
    { x: 1110, y: 392 },
    62,
    { yaw: angles.yaw + 0.08, pitch: angles.pitch + 0.03, roll: angles.roll },
    focusIndex,
    1,
  );
  const vSpace = buildProjectedSpace(
    vectorSpaces.vAll,
    { x: 930, y: 548 },
    62,
    { yaw: angles.yaw - 0.05, pitch: angles.pitch - 0.02, roll: angles.roll },
    focusIndex,
    2,
  );
  const headSpaces = [
    buildProjectedSpace(
      vectorSpaces.heads[0],
      { x: 438, y: 236 },
      56,
      angles,
      focusIndex,
      0,
    ),
    buildProjectedSpace(
      vectorSpaces.heads[1],
      { x: 438, y: 392 },
      56,
      { yaw: angles.yaw + 0.04, pitch: angles.pitch, roll: angles.roll },
      focusIndex,
      1,
    ),
    buildProjectedSpace(
      vectorSpaces.heads[2],
      { x: 438, y: 548 },
      56,
      { yaw: angles.yaw - 0.06, pitch: angles.pitch, roll: angles.roll },
      focusIndex,
      2,
    ),
  ];
  const mixSpace = buildProjectedSpace(
    vectorSpaces.mix,
    { x: 1030, y: 392 },
    88,
    angles,
    focusIndex,
    0,
  );
  const outputInputSpace = buildProjectedSpace(
    vectorSpaces.input,
    { x: 360, y: 392 },
    70,
    angles,
    focusIndex,
    0,
  );
  const outputMixSpace = buildProjectedSpace(
    vectorSpaces.mix,
    { x: 724, y: 392 },
    70,
    angles,
    focusIndex,
    1,
  );
  const outputSpace = buildProjectedSpace(
    vectorSpaces.output,
    { x: 1114, y: 392 },
    92,
    angles,
    focusIndex,
    2,
  );

  const tokenY = TOKENS.map((_, index) => 288 + index * 72);
  const stageOffset = (opacity: number, amount = 24) => (1 - opacity) * amount;
  const sceneTransform = (opacity: number, amount = 24, depth = 1) =>
    `translate(${stageOffset(opacity, amount) - scrollLift * 18 * depth} ${
      scrollLift * 32 * depth
    })`;
  const ambientDots = Array.from({ length: 16 }, (_, index) => ({
    x: 120 + index * 78,
    y:
      120 +
      (index % 2) * 92 +
      Math.sin(index * 0.8) * 82 +
      (reducedMotion ? 0 : Math.sin(elapsed * 0.00025 + index) * 20) +
      scrollLift * (22 + (index % 3) * 4),
    r: 0.8 + (index % 3) * 0.45,
    opacity: 0.12 + (index % 4) * 0.045,
  }));

  const stageStyle = reducedMotion
    ? undefined
    : {
        transform: `perspective(1800px) translate3d(0, ${
          scrollLift * 42
        }px, 0) scale(${0.928 + scrollDepth * 0.072}) rotateX(${
          scrollLift * 10
        }deg)`,
        opacity: 0.72 + scrollDepth * 0.28,
        filter: `drop-shadow(0 36px 72px rgba(0, 0, 0, ${
          0.18 + scrollLift * 0.2
        }))`,
      };

  const svgStyle = reducedMotion
    ? undefined
    : {
        transform: `translate3d(0, ${scrollLift * 12}px, 0) scale(${
          0.986 + scrollDepth * 0.014
        })`,
      };

  const metaStyle = reducedMotion
    ? undefined
    : {
        transform: `translate3d(0, ${scrollLift * 18}px, 0)`,
        opacity: 0.7 + scrollDepth * 0.3,
      };

  const captionStyle = reducedMotion
    ? undefined
    : {
        transform: `translate3d(0, ${scrollLift * 20}px, 0)`,
        opacity: 0.66 + scrollDepth * 0.34,
      };

  const svgLabel = `AI flow shown as five stage screens. Focus word ${focusToken} moves from input context to Q K V projection, attention weighting, context mixing, and experience-ready output.`;

  return (
    <div ref={rootRef} className={styles.root}>
      <div
        className={styles.stage}
        style={stageStyle}
        role="img"
        aria-label={svgLabel}
      >
        <svg
          className={styles.svg}
          style={svgStyle}
          viewBox="0 0 1440 760"
          xmlns="http://www.w3.org/2000/svg"
        >
          <defs>
            <pattern
              id={`${ids}-grid`}
              width="36"
              height="36"
              patternUnits="userSpaceOnUse"
            >
              <path
                d="M 36 0 L 0 0 0 36"
                fill="none"
                stroke="rgba(255,255,255,0.038)"
                strokeWidth="1"
              />
            </pattern>
            <radialGradient id={`${ids}-amber`} cx="28%" cy="34%" r="54%">
              <stop offset="0%" stopColor="rgba(242,196,109,0.18)" />
              <stop offset="100%" stopColor="rgba(242,196,109,0)" />
            </radialGradient>
            <radialGradient id={`${ids}-blue`} cx="54%" cy="46%" r="54%">
              <stop offset="0%" stopColor="rgba(110,182,255,0.16)" />
              <stop offset="100%" stopColor="rgba(110,182,255,0)" />
            </radialGradient>
            <radialGradient id={`${ids}-green`} cx="76%" cy="62%" r="48%">
              <stop offset="0%" stopColor="rgba(116,223,186,0.16)" />
              <stop offset="100%" stopColor="rgba(116,223,186,0)" />
            </radialGradient>
          </defs>

          <rect
            x="0"
            y="0"
            width="1440"
            height="760"
            fill={`url(#${ids}-grid)`}
            opacity="0.42"
          />
          <ellipse
            cx="430"
            cy={320 + scrollLift * 24}
            rx="320"
            ry="220"
            fill={`url(#${ids}-amber)`}
            opacity={0.1 + stageMap.embed.opacity * 0.4 + stageMap.qkv.opacity * 0.18}
          />
          <ellipse
            cx="760"
            cy={396 + scrollLift * 14}
            rx="340"
            ry="240"
            fill={`url(#${ids}-blue)`}
            opacity={0.08 + stageMap.attention.opacity * 0.4 + stageMap.qkv.opacity * 0.18}
          />
          <ellipse
            cx="1110"
            cy={440 + scrollLift * 8}
            rx="290"
            ry="210"
            fill={`url(#${ids}-green)`}
            opacity={0.08 + stageMap.heads.opacity * 0.34 + stageMap.output.opacity * 0.22}
          />

          {ambientDots.map((dot, index) => (
            <circle
              key={index}
              cx={dot.x}
              cy={dot.y}
              r={dot.r}
              fill="rgba(245,247,251,0.9)"
              opacity={dot.opacity}
            />
          ))}

          <text
            x="56"
            y="68"
            fontFamily="JetBrains Mono, monospace"
            fontSize="13"
            letterSpacing="0.24em"
            fill="rgba(245,247,251,0.52)"
          >
            AI FLOW · CONTEXT TO EXPERIENCE
          </text>
          <text
            x="1382"
            y="68"
            textAnchor="end"
            fontFamily="JetBrains Mono, monospace"
            fontSize="12"
            letterSpacing="0.16em"
            fill="rgba(245,247,251,0.38)"
          >
            {`FOCUS WORD · ${focusToken}`}
          </text>

          <g transform={`translate(0 ${scrollLift * 12})`}>
            {STAGES.map((stage, index) => {
              const active = stage.id === currentStage.id;
              return (
                <g key={stage.id}>
                  <line
                    x1={224 + index * 204}
                    y1="98"
                    x2={380 + index * 204}
                    y2="98"
                    stroke="rgba(255,255,255,0.08)"
                    strokeWidth="1"
                    opacity={index === STAGES.length - 1 ? 0 : 1}
                  />
                  <circle
                    cx={180 + index * 204}
                    cy="98"
                    r={active ? 9 : 6}
                    fill={active ? "rgba(245,247,251,0.94)" : "rgba(255,255,255,0.18)"}
                  />
                  <text
                    x={180 + index * 204}
                    y="126"
                    textAnchor="middle"
                    fontFamily="JetBrains Mono, monospace"
                    fontSize="11"
                    letterSpacing="0.16em"
                    fill={active ? "rgba(245,247,251,0.9)" : "rgba(245,247,251,0.44)"}
                  >
                    {stage.step}
                  </text>
                </g>
              );
            })}
          </g>

          <g transform={`translate(0 ${scrollLift * 18})`}>
            <text
              x="56"
              y="156"
              fontFamily="JetBrains Mono, monospace"
              fontSize="14"
              letterSpacing="0.2em"
              fill="rgba(245,247,251,0.84)"
            >
              {currentStage.step}
            </text>
            <text
              x="56"
              y="194"
              fontFamily="DM Sans, system-ui, sans-serif"
              fontSize="32"
              fontWeight="700"
              fill="rgba(245,247,251,0.96)"
            >
              {currentStage.title}
            </text>
            <text
              x="56"
              y="224"
              fontFamily="JetBrains Mono, monospace"
              fontSize="13"
              letterSpacing="0.14em"
              fill="rgba(245,247,251,0.48)"
            >
              {currentStage.formula}
            </text>
          </g>

          <g
            opacity={stageMap.embed.opacity}
            transform={sceneTransform(stageMap.embed.opacity, 28, 1.14)}
          >
            {TOKENS.map((token, index) => (
              <g key={token}>
                <rect
                  x="76"
                  y={tokenY[index] - 24}
                  width="102"
                  height="48"
                  rx="24"
                  fill={
                    index === focusIndex
                      ? "rgba(255,255,255,0.06)"
                      : "rgba(255,255,255,0.022)"
                  }
                  stroke={
                    index === focusIndex
                      ? "rgba(255,255,255,0.28)"
                      : "rgba(255,255,255,0.08)"
                  }
                />
                <text
                  x="127"
                  y={tokenY[index] + 7}
                  textAnchor="middle"
                  fontFamily="DM Sans, system-ui, sans-serif"
                  fontSize="24"
                  fontWeight="700"
                  fill="rgba(245,247,251,0.94)"
                >
                  {token}
                </text>
                <path
                  d={positionalWavePath(
                    212,
                    tokenY[index] - 18,
                    64,
                    36,
                    progress * Math.PI * 2 + index * 0.54,
                  )}
                  fill="none"
                  stroke={index === focusIndex ? Q_COLOR : "rgba(245,247,251,0.34)"}
                  strokeWidth="1.15"
                  opacity={0.36}
                />
                <path
                  d={curvePath(178, tokenY[index], 330, tokenY[index], 0.28)}
                  fill="none"
                  stroke="rgba(255,255,255,0.14)"
                  strokeWidth="1.2"
                  strokeDasharray="5 10"
                  opacity="0.5"
                />
              </g>
            ))}
            <path
              d={curvePath(332, tokenY[focusIndex], embedSpace.focus.x - 20, embedSpace.focus.y, 0.36)}
              fill="none"
              stroke="rgba(245,247,251,0.4)"
              strokeWidth="1.5"
              strokeDasharray="5 9"
              opacity="0.68"
            />
            <SpaceCluster
              space={embedSpace}
              label="CONTEXT SPACE"
              color={OUTPUT_COLOR}
              opacity={1}
              haloRadius={138}
              planeColor="rgba(245,247,251,0.5)"
            />
          </g>

          <g
            opacity={stageMap.qkv.opacity}
            transform={sceneTransform(stageMap.qkv.opacity, 28, 1.08)}
          >
            <SpaceCluster
              space={projectionInputSpace}
              label="SOURCE INPUT"
              color={OUTPUT_COLOR}
              opacity={1}
              haloRadius={92}
              labelDy={-78}
            />
            <SpaceCluster
              space={qSpace}
              label="QUERY"
              color={Q_COLOR}
              opacity={1}
              haloRadius={72}
              labelDy={-68}
            />
            <SpaceCluster
              space={kSpace}
              label="KEY"
              color={K_COLOR}
              opacity={1}
              haloRadius={72}
              labelDy={-68}
            />
            <SpaceCluster
              space={vSpace}
              label="VALUE"
              color={V_COLOR}
              opacity={1}
              haloRadius={72}
              labelDy={-68}
            />
            {[qSpace, kSpace, vSpace].map((space, index) => (
              <path
                key={index}
                d={curvePath(
                  projectionInputSpace.focus.x + 18,
                  projectionInputSpace.focus.y,
                  space.focus.x - 18,
                  space.focus.y,
                  0.34,
                )}
                fill="none"
                stroke={[Q_COLOR, K_COLOR, V_COLOR][index]}
                strokeWidth="1.5"
                strokeDasharray="5 9"
                opacity="0.72"
              />
            ))}
          </g>

          <g
            opacity={stageMap.attention.opacity}
            transform={sceneTransform(stageMap.attention.opacity, 20, 0.96)}
          >
            <rect
              x="90"
              y={tokenY[focusIndex] - 24}
              width="112"
              height="48"
              rx="24"
              fill="rgba(255,255,255,0.05)"
              stroke="rgba(255,255,255,0.18)"
            />
            <text
              x="146"
              y={tokenY[focusIndex] + 7}
              textAnchor="middle"
              fontFamily="DM Sans, system-ui, sans-serif"
              fontSize="24"
              fontWeight="700"
              fill="rgba(245,247,251,0.94)"
            >
              {focusToken}
            </text>
            <text
              x="146"
              y={tokenY[focusIndex] - 42}
              textAnchor="middle"
              fontFamily="JetBrains Mono, monospace"
              fontSize="12"
              letterSpacing="0.16em"
              fill="rgba(245,247,251,0.48)"
            >
              FOCUS WORD
            </text>

            {pipeline.attention.map((matrix, headIndex) => {
              const x = 542 + headIndex * 260;
              const y = 214;
              return (
                <g key={headIndex}>
                  <path
                    d={curvePath(220, tokenY[focusIndex], x - 28, y + 84, 0.44)}
                    fill="none"
                    stroke={HEAD_COLORS[headIndex]}
                    strokeWidth="1.4"
                    strokeDasharray="4 9"
                    opacity="0.68"
                  />
                  <AttentionMatrix
                    x={x}
                    y={y}
                    size={168}
                    matrix={matrix}
                    focusIndex={focusIndex}
                    color={HEAD_COLORS[headIndex]}
                    label={`HEAD 0${headIndex + 1}`}
                    opacity={1}
                  />
                </g>
              );
            })}
          </g>

          <g
            opacity={stageMap.heads.opacity}
            transform={sceneTransform(stageMap.heads.opacity, 24, 1.02)}
          >
            {headSpaces.map((space, index) => (
              <g key={index}>
                <SpaceCluster
                  space={space}
                  label={`HEAD 0${index + 1}`}
                  color={HEAD_COLORS[index]}
                  opacity={1}
                  haloRadius={60}
                  labelDy={-66}
                />
                <path
                  d={curvePath(space.focus.x + 18, space.focus.y, mixSpace.focus.x - 24, mixSpace.focus.y, 0.34)}
                  fill="none"
                  stroke={HEAD_COLORS[index]}
                  strokeWidth="1.5"
                  strokeDasharray="5 9"
                  opacity="0.72"
                />
              </g>
            ))}
            <SpaceCluster
              space={mixSpace}
              label="CONCATENATED CONTEXT"
              color={OUTPUT_COLOR}
              opacity={1}
              haloRadius={106}
              labelDy={-82}
            />
          </g>

          <g
            opacity={stageMap.output.opacity}
            transform={sceneTransform(stageMap.output.opacity, 20, 0.9)}
          >
            <SpaceCluster
              space={outputInputSpace}
              label="RESIDUAL INPUT"
              color={OUTPUT_COLOR}
              opacity={1}
              haloRadius={76}
              labelDy={-70}
            />
            <text
              x="536"
              y="402"
              textAnchor="middle"
              fontFamily="DM Sans, system-ui, sans-serif"
              fontSize="44"
              fontWeight="700"
              fill="rgba(245,247,251,0.8)"
            >
              +
            </text>
            <SpaceCluster
              space={outputMixSpace}
              label="MHA OUTPUT"
              color={K_COLOR}
              opacity={1}
              haloRadius={76}
              labelDy={-70}
            />
            <path
              d={curvePath(outputInputSpace.focus.x + 18, outputInputSpace.focus.y, outputMixSpace.focus.x - 18, outputMixSpace.focus.y, 0.34)}
              fill="none"
              stroke="rgba(245,247,251,0.28)"
              strokeWidth="1.35"
              strokeDasharray="5 10"
              opacity="0.62"
            />
            <path
              d={curvePath(outputMixSpace.focus.x + 20, outputMixSpace.focus.y, outputSpace.focus.x - 20, outputSpace.focus.y, 0.34)}
              fill="none"
              stroke={OUTPUT_COLOR}
              strokeWidth="1.5"
              strokeDasharray="5 9"
              opacity="0.74"
            />
            <text
              x="908"
              y="332"
              textAnchor="middle"
              fontFamily="JetBrains Mono, monospace"
              fontSize="12"
              letterSpacing="0.18em"
              fill="rgba(245,247,251,0.58)"
            >
              LAYERNORM → FFN
            </text>
            <SpaceCluster
              space={outputSpace}
              label="EXPERIENCE-READY OUTPUT"
              color={OUTPUT_COLOR}
              opacity={1}
              haloRadius={112}
              labelDy={-86}
              planeColor="rgba(116,223,186,0.48)"
            />
          </g>
        </svg>
      </div>

      <div className={styles.metaRow} style={metaStyle}>
        <span className={styles.metaChip} title={currentStage.step}>
          {currentStage.step}
        </span>
        <span className={styles.metaChip} title={currentStage.title}>
          {currentStage.title}
        </span>
        <span className={styles.metaChip} title={currentStage.formula}>
          {currentStage.formula}
        </span>
      </div>
      <p className={styles.caption} style={captionStyle}>
        {currentStage.summary}
      </p>
    </div>
  );
}
