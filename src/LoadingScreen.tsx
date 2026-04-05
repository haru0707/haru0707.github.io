import { useEffect, useRef, useState } from "react";
import { ThreeBodyCanvas } from "./ThreeBodyCanvas";
import styles from "./LoadingScreen.module.css";

type Props = {
  onComplete?: () => void;
  minMs?: number;
  exiting?: boolean;
};

export function LoadingScreen({
  onComplete,
  minMs = 3200,
  exiting = false,
}: Props) {
  const [progress, setProgress] = useState(0);
  const onCompleteRef = useRef(onComplete);
  const completedRef = useRef(false);
  onCompleteRef.current = onComplete;

  useEffect(() => {
    const start = performance.now();
    let raf = 0;
    completedRef.current = false;
    let completionTimer = 0;

    const finish = () => {
      if (completedRef.current) return;
      completedRef.current = true;
      setProgress(1);
      onCompleteRef.current?.();
    };

    const tick = (now: number) => {
      if (completedRef.current) return;
      const t = Math.min(1, (now - start) / minMs);
      setProgress(t);
      if (t >= 1) return;
      raf = requestAnimationFrame(tick);
    };

    completionTimer = window.setTimeout(finish, minMs);
    raf = requestAnimationFrame(tick);

    return () => {
      completedRef.current = true;
      cancelAnimationFrame(raf);
      if (completionTimer) window.clearTimeout(completionTimer);
    };
  }, [minMs]);

  return (
    <div
      className={`${styles.root} ${exiting ? styles.exit : ""}`}
      role="status"
      aria-live="polite"
      aria-label="読み込み中"
    >
      <div className={styles.canvasWrap}>
        <ThreeBodyCanvas active={!exiting} />
      </div>
      <div className={styles.overlay}>
        <div className={styles.cornerTL}>
          <span className="mono">SYS.INIT</span>
        </div>
        <div className={styles.cornerBR}>
          <span className="mono">N=3 · APPROX</span>
        </div>
        <div className={styles.centerBlock}>
          <p className={styles.kicker}>THREE-BODY · NUMERIC</p>
          <h1 className={styles.title}>LOADING</h1>
          <div className={styles.barTrack}>
            <div
              className={styles.barFill}
              style={{ transform: `scaleX(${progress})` }}
            />
          </div>
          <p className={`${styles.pct} mono`}>
            {Math.round(progress * 100).toString().padStart(3, "0")}%
          </p>
        </div>
      </div>
    </div>
  );
}
