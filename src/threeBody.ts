/** 2D 三体問題の数値積分（ソフトニング付き重力） */

export type Vec2 = [number, number];

export interface ThreeBodyState {
  masses: [number, number, number];
  pos: [Vec2, Vec2, Vec2];
  vel: [Vec2, Vec2, Vec2];
}

const G = 1.2;
const SOFT = 0.06;

function accelOn(
  i: 0 | 1 | 2,
  pos: [Vec2, Vec2, Vec2],
  masses: [number, number, number]
): Vec2 {
  let ax = 0;
  let ay = 0;
  for (let j = 0; j < 3; j++) {
    if (j === i) continue;
    const dx = pos[j][0] - pos[i][0];
    const dy = pos[j][1] - pos[i][1];
    const r2 = dx * dx + dy * dy + SOFT * SOFT;
    const invR = 1 / Math.sqrt(r2);
    const invR3 = invR * invR * invR;
    const f = G * masses[j] * invR3;
    ax += f * dx;
    ay += f * dy;
  }
  return [ax, ay];
}

/** 速度ベルレ法（1ステップ） */
export function stepVerlet(
  state: ThreeBodyState,
  dt: number,
  substeps: number
): ThreeBodyState {
  const { masses } = state;
  let pos: [Vec2, Vec2, Vec2] = [
    [...state.pos[0]] as Vec2,
    [...state.pos[1]] as Vec2,
    [...state.pos[2]] as Vec2,
  ];
  let vel: [Vec2, Vec2, Vec2] = [
    [...state.vel[0]] as Vec2,
    [...state.vel[1]] as Vec2,
    [...state.vel[2]] as Vec2,
  ];

  const h = dt / substeps;
  for (let s = 0; s < substeps; s++) {
    const a0 = [
      accelOn(0, pos, masses),
      accelOn(1, pos, masses),
      accelOn(2, pos, masses),
    ] as [Vec2, Vec2, Vec2];

    for (let i = 0; i < 3; i++) {
      pos[i][0] += vel[i][0] * h + 0.5 * a0[i][0] * h * h;
      pos[i][1] += vel[i][1] * h + 0.5 * a0[i][1] * h * h;
    }

    const a1 = [
      accelOn(0, pos, masses),
      accelOn(1, pos, masses),
      accelOn(2, pos, masses),
    ] as [Vec2, Vec2, Vec2];

    for (let i = 0; i < 3; i++) {
      vel[i][0] += 0.5 * (a0[i][0] + a1[i][0]) * h;
      vel[i][1] += 0.5 * (a0[i][1] + a1[i][1]) * h;
    }
  }

  return { masses, pos, vel };
}

/** バリセンター周りに正規化したスケール付き座標（描画用） */
export function barycentricFrame(
  pos: [Vec2, Vec2, Vec2],
  masses: [number, number, number]
): { shifted: [Vec2, Vec2, Vec2]; scale: number } {
  const M = masses[0] + masses[1] + masses[2];
  let cx = 0;
  let cy = 0;
  for (let i = 0; i < 3; i++) {
    cx += masses[i] * pos[i][0];
    cy += masses[i] * pos[i][1];
  }
  cx /= M;
  cy /= M;

  const shifted: [Vec2, Vec2, Vec2] = [
    [pos[0][0] - cx, pos[0][1] - cy],
    [pos[1][0] - cx, pos[1][1] - cy],
    [pos[2][0] - cx, pos[2][1] - cy],
  ];

  let maxD = 0.001;
  for (let i = 0; i < 3; i++) {
    const d = Math.hypot(shifted[i][0], shifted[i][1]);
    if (d > maxD) maxD = d;
  }
  return { shifted, scale: 1 / maxD };
}

/**
 * カオス的だが比較的バウンドしやすい初期条件（等質量・対称性を崩す）
 */
export function createInitialState(): ThreeBodyState {
  const masses: [number, number, number] = [1, 1.05, 0.95];
  const pos: [Vec2, Vec2, Vec2] = [
    [0.88, 0.12],
    [-0.76, -0.38],
    [-0.1, 0.72],
  ];
  const vel: [Vec2, Vec2, Vec2] = [
    [-0.22, 0.42],
    [0.35, -0.18],
    [-0.18, -0.28],
  ];
  return { masses, pos, vel };
}
