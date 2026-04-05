import { useEffect, useRef } from "react";
import {
  barycentricFrame,
  createInitialState,
  stepVerlet,
  type Vec2,
} from "./threeBody";

const TRAIL_LEN = 420;
const COLORS = ["#ffffff", "#e0e0e0", "#b8b8b8"] as const;

type Props = {
  active: boolean;
};

export function ThreeBodyCanvas({ active }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stateRef = useRef(createInitialState());
  const trailsRef = useRef<[Vec2[], Vec2[], Vec2[]]>([[], [], []]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let raf = 0;
    let last = performance.now();

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio ?? 1, 2);
      const w = window.innerWidth;
      const h = window.innerHeight;
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    window.addEventListener("resize", resize);

    const draw = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;

      if (active) {
        stateRef.current = stepVerlet(stateRef.current, dt * 1.15, 8);
        const { pos, masses } = stateRef.current;
        const { shifted } = barycentricFrame(pos, masses);
        const trails = trailsRef.current;
        for (let i = 0; i < 3; i++) {
          const arr = trails[i];
          arr.push([shifted[i][0], shifted[i][1]]);
          if (arr.length > TRAIL_LEN) arr.shift();
        }
      }

      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      const cx = w * 0.5;
      const cy = h * 0.48;
      const baseR = Math.min(w, h) * 0.22;

      ctx.fillStyle = "#050505";
      ctx.fillRect(0, 0, w, h);

      const { pos, masses } = stateRef.current;
      const { shifted, scale } = barycentricFrame(pos, masses);
      const trails = trailsRef.current;

      ctx.save();
      ctx.translate(cx, cy);
      ctx.strokeStyle = "rgba(255,255,255,0.06)";
      ctx.lineWidth = 1;
      const rings = 5;
      for (let r = 1; r <= rings; r++) {
        ctx.beginPath();
        ctx.arc(0, 0, (baseR * r) / rings, 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.restore();

      for (let i = 0; i < 3; i++) {
        const arr = trails[i];
        if (arr.length < 2) continue;
        ctx.beginPath();
        ctx.strokeStyle = `${COLORS[i]}22`;
        ctx.lineWidth = 1;
        for (let k = 0; k < arr.length; k++) {
          const sx = cx + arr[k][0] * scale * baseR * 0.92;
          const sy = cy + arr[k][1] * scale * baseR * 0.92;
          if (k === 0) ctx.moveTo(sx, sy);
          else ctx.lineTo(sx, sy);
        }
        ctx.stroke();

        ctx.beginPath();
        ctx.strokeStyle = `${COLORS[i]}55`;
        ctx.lineWidth = 1.2;
        const start = Math.max(0, arr.length - 80);
        for (let k = start; k < arr.length; k++) {
          const sx = cx + arr[k][0] * scale * baseR * 0.92;
          const sy = cy + arr[k][1] * scale * baseR * 0.92;
          if (k === start) ctx.moveTo(sx, sy);
          else ctx.lineTo(sx, sy);
        }
        ctx.stroke();
      }

      for (let i = 0; i < 3; i++) {
        const sx = cx + shifted[i][0] * scale * baseR * 0.92;
        const sy = cy + shifted[i][1] * scale * baseR * 0.92;
        const rad = 3 + masses[i] * 2.2;
        ctx.beginPath();
        ctx.fillStyle = COLORS[i];
        ctx.arc(sx, sy, rad, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#050505";
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      raf = requestAnimationFrame(draw);
    };

    raf = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
    };
  }, [active]);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden
      style={{ display: "block", width: "100%", height: "100%" }}
    />
  );
}
