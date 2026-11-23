import {
  type CSSProperties,
  useEffect,
  useRef,
  useState,
} from "react";
import { LoadingScreen } from "./LoadingScreen";
import { AttentionHero } from "./AttentionHero";
import styles from "./App.module.css";

const socialLinks = [
  {
    label: "Kaggle",
    href: "https://www.kaggle.com/haru0707",
  },
  {
    label: "GitHub",
    href: "https://github.com/haru0707",
  },
  {
    label: "X",
    href: "https://x.com/haru0707kun",
  },
] as const;

const BOOT_MIN_MS = 1000;
const BOOT_MAX_MS = 2600;
const KAGGLE_URL = "https://www.kaggle.com/haru0707";

const heroStats = [
  {
    label: "JOAI 2026",
    value: "銅賞",
    sub: "日本人工知能オリンピック",
    accent: "var(--gold)",
    href: "#results",
  },
  {
    label: "KAGGLE",
    value: "Expert",
    sub: "Silver ×2 · Bronze ×3",
    accent: "var(--blue)",
    href: "#kaggle",
  },
  {
    label: "ICFPC 2026",
    value: "16",
    unit: "位",
    sub: "ICFP Programming Contest",
    accent: "var(--mint)",
    href: "#results",
  },
  {
    label: "BEST FINISH",
    value: "21",
    unit: "/ 2,963",
    sub: "NeuroGolf 2026 · Top 0.7%",
    accent: "#ffffff",
    href: "#kaggle",
  },
] as const;

const awards = [
  {
    year: "2026",
    title: "日本人工知能オリンピック（JOAI 2026）",
    desc: "AI・機械学習の課題に取り組む競技大会で銅賞を受賞。",
    result: "銅賞",
    accent: "var(--gold)",
  },
  {
    year: "2026",
    title: "ICFP Programming Contest 2026",
    desc: "国際会議 ICFP に合わせて開かれる、長時間のプログラミングコンテスト。",
    result: "16位",
    accent: "var(--mint)",
  },
  {
    year: "2026",
    title: "Kaggle Competitions Expert",
    desc: "銀メダル 2 個・銅メダル 3 個を獲得。Competitions ランキング最高 1,867 位（約 22.4 万人中）。",
    result: "Expert",
    accent: "var(--blue)",
  },
  {
    year: "U16",
    title: "U16プログラミングコンテスト 北海道大会",
    desc: "ロジックと実装力の両方が問われる場で準優勝。",
    result: "準優勝",
    accent: "#ffffff",
  },
] as const;

const kaggleMedals = [
  {
    medal: "silver",
    name: "The 2026 NeuroGolf Championship",
    note: "ARC-AGI の画像変換を解く最小のニューラルネットを設計",
    rank: 21,
    teams: 2963,
    href: "https://www.kaggle.com/competitions/neurogolf-2026",
  },
  {
    medal: "silver",
    name: "BirdCLEF+ 2026",
    note: "パンタナールの音声から生物種を識別",
    rank: 150,
    teams: 4094,
    href: "https://www.kaggle.com/competitions/birdclef-2026",
  },
  {
    medal: "bronze",
    name: "The Pokémon Company - PTCG AI Battle Challenge",
    note: "ポケモンカードゲームをプレイする AI エージェント",
    rank: 503,
    teams: 6807,
    href: "https://www.kaggle.com/competitions/pokemon-tcg-ai-battle",
  },
  {
    medal: "bronze",
    name: "Kaggriculture",
    note: "農業シミュレーションで収益を最大化するエージェント",
    rank: 813,
    teams: 10148,
    href: "https://www.kaggle.com/competitions/kaggriculture",
  },
  {
    medal: "bronze",
    name: "Orbit Wars",
    note: "連続 2D 空間のリアルタイムストラテジー",
    rank: 471,
    teams: 4729,
    href: "https://www.kaggle.com/competitions/orbit-wars",
  },
] as const;

const BIRDCLEF_WRITEUP_URL =
  "https://www.kaggle.com/competitions/birdclef-2026/writeups/149th-place-solution-private-0-94352";

const numberFormat = new Intl.NumberFormat("en-US");

function formatTopPercent(rank: number, teams: number) {
  const percent = (rank / teams) * 100;
  return `TOP ${percent < 10 ? percent.toFixed(1) : Math.ceil(percent)}%`;
}

const profileCapabilities = [
  {
    id: "01",
    icon: "interface",
    title: "AIとUIの統合",
    desc: "AI の機能を UI に自然になじませ、体験として分かりやすく見せる設計ができます。",
  },
  {
    id: "02",
    icon: "network",
    title: "DLエンジニアリング",
    desc: "DL や LLM を独学で深めながら、検証から実装まで粘り強く進められます。",
  },
  {
    id: "03",
    icon: "code",
    title: "Webサイト作成",
    desc: "雰囲気を崩さずに、設計・実装・改善・公開まで一貫して対応できます。",
  },
] as const;

function Arrow() {
  return (
    <svg className={styles.arrow} viewBox="0 0 12 12" fill="none" aria-hidden>
      <path d="M2.5 9.5 9.5 2.5M4 2.5h5.5V8" stroke="currentColor" strokeWidth="1.4" />
    </svg>
  );
}

function CapabilityIcon({ name }: { name: "interface" | "network" | "code" }) {
  return (
    <svg
      className={styles.workCapabilityIcon}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.4"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      {name === "interface" ? (
        <>
          <rect x="3" y="4.5" width="18" height="15" rx="2.5" />
          <path d="M3 8.5h18" />
          <path d="m12 11 .9 2.1 2.1.9-2.1.9-.9 2.1-.9-2.1-2.1-.9 2.1-.9z" />
        </>
      ) : name === "network" ? (
        <>
          <path d="M5 7.5 12 5M5 7.5l7 4.5M5 7.5 12 19M5 16.5 12 5M5 16.5l7-4.5M5 16.5l7 2.5M12 5l7 7M12 12h7M12 19l7-7" opacity="0.55" />
          <circle cx="5" cy="7.5" r="1.6" />
          <circle cx="5" cy="16.5" r="1.6" />
          <circle cx="12" cy="5" r="1.6" />
          <circle cx="12" cy="12" r="1.6" />
          <circle cx="12" cy="19" r="1.6" />
          <circle cx="19" cy="12" r="1.6" />
        </>
      ) : (
        <>
          <path d="m8.5 8-4 4 4 4" />
          <path d="m15.5 8 4 4-4 4" />
          <path d="m13.5 5.5-3 13" />
        </>
      )}
    </svg>
  );
}

function clamp01(value: number) {
  return Math.max(0, Math.min(1, value));
}

function easeInOutCubic(value: number) {
  return value < 0.5
    ? 4 * value * value * value
    : 1 - Math.pow(-2 * value + 2, 3) / 2;
}

function calculateAge(birthDate: string, now = new Date()) {
  const birth = new Date(`${birthDate}T00:00:00`);
  let age = now.getFullYear() - birth.getFullYear();
  const monthDiff = now.getMonth() - birth.getMonth();
  const dayDiff = now.getDate() - birth.getDate();

  if (monthDiff < 0 || (monthDiff === 0 && dayDiff < 0)) {
    age -= 1;
  }

  return age;
}

function useElementHeight<T extends HTMLElement>() {
  const elementRef = useRef<T>(null);
  const [height, setHeight] = useState(0);

  useEffect(() => {
    const element = elementRef.current;
    if (!element) return;

    let raf = 0;

    const measure = () => {
      raf = 0;
      const nextHeight = element.getBoundingClientRect().height;
      setHeight((currentHeight) =>
        Math.abs(currentHeight - nextHeight) < 0.5 ? currentHeight : nextHeight,
      );
    };

    const scheduleMeasure = () => {
      if (!raf) raf = requestAnimationFrame(measure);
    };

    measure();

    let observer: ResizeObserver | undefined;
    if ("ResizeObserver" in window) {
      observer = new ResizeObserver(scheduleMeasure);
      observer.observe(element);
    }

    window.addEventListener("resize", scheduleMeasure);

    return () => {
      if (raf) cancelAnimationFrame(raf);
      observer?.disconnect();
      window.removeEventListener("resize", scheduleMeasure);
    };
  }, []);

  return [elementRef, height] as const;
}

function useViewportHeight() {
  const [viewportHeight, setViewportHeight] = useState(0);

  useEffect(() => {
    let raf = 0;

    const measure = () => {
      raf = 0;
      const nextHeight = window.visualViewport?.height ?? window.innerHeight;
      setViewportHeight((currentHeight) =>
        Math.abs(currentHeight - nextHeight) < 0.5 ? currentHeight : nextHeight,
      );
    };

    const scheduleMeasure = () => {
      if (!raf) raf = requestAnimationFrame(measure);
    };

    measure();

    window.addEventListener("resize", scheduleMeasure);
    window.visualViewport?.addEventListener("resize", scheduleMeasure);
    window.visualViewport?.addEventListener("scroll", scheduleMeasure);

    return () => {
      if (raf) cancelAnimationFrame(raf);
      window.removeEventListener("resize", scheduleMeasure);
      window.visualViewport?.removeEventListener("resize", scheduleMeasure);
      window.visualViewport?.removeEventListener("scroll", scheduleMeasure);
    };
  }, []);

  return viewportHeight;
}

function useScrollReveal(enabled: boolean) {
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!enabled) return;

    const root = rootRef.current;
    if (!root) return;

    const elements = Array.from(root.querySelectorAll<HTMLElement>("[data-reveal]"));
    if (!elements.length) return;

    elements.forEach((element) => {
      const delay = Number(element.dataset.revealDelay ?? 0);
      element.dataset.revealState = "hidden";
      element.style.setProperty("--reveal-delay", `${delay}ms`);
    });

    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    if (media.matches) {
      elements.forEach((element) => {
        element.dataset.revealState = "visible";
      });
      return;
    }

    const revealIfVisible = (element: HTMLElement) => {
      const rect = element.getBoundingClientRect();
      const viewportHeight = window.innerHeight || 1;
      if (rect.top <= viewportHeight * 0.9 && rect.bottom >= viewportHeight * 0.12) {
        element.dataset.revealState = "visible";
        return true;
      }
      return false;
    };

    if (!("IntersectionObserver" in window)) {
      elements.forEach((element) => {
        element.dataset.revealState = "visible";
      });
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          entry.target.setAttribute("data-reveal-state", "visible");
          observer.unobserve(entry.target);
        });
      },
      {
        threshold: 0.16,
        rootMargin: "0px 0px -12% 0px",
      },
    );

    elements.forEach((element) => {
      if (revealIfVisible(element)) return;
      observer.observe(element);
    });

    return () => observer.disconnect();
  }, [enabled]);

  return rootRef;
}

function useBootReady(minMs: number, maxMs: number) {
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let minElapsed = false;
    let pageLoaded = document.readyState === "complete";
    let fontsLoaded = !("fonts" in document);
    let timer = 0;
    let raf = 0;

    const finishIfReady = () => {
      if (cancelled || !minElapsed || !pageLoaded || !fontsLoaded) return;
      if (raf) cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        if (!cancelled) setReady(true);
      });
    };

    timer = window.setTimeout(() => {
      minElapsed = true;
      finishIfReady();
    }, minMs);

    // Never hold the page behind slow fonts or images: text swaps in later.
    const giveUp = window.setTimeout(() => {
      pageLoaded = true;
      fontsLoaded = true;
      finishIfReady();
    }, maxMs);

    const onLoad = () => {
      pageLoaded = true;
      finishIfReady();
    };

    if (!pageLoaded) {
      window.addEventListener("load", onLoad, { once: true });
    }

    const fontSet = (document as Document & {
      fonts?: { ready?: Promise<unknown> };
    }).fonts;

    if (fontSet?.ready) {
      fontSet.ready
        .catch(() => undefined)
        .then(() => {
          fontsLoaded = true;
          finishIfReady();
        });
    } else {
      fontsLoaded = true;
    }

    finishIfReady();

    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
      window.clearTimeout(giveUp);
      if (raf) cancelAnimationFrame(raf);
      window.removeEventListener("load", onLoad);
    };
  }, [minMs, maxMs]);

  return ready;
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

function usePinnedSectionProgress(enabled: boolean, reducedMotion: boolean) {
  const sectionRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const element = sectionRef.current;
    if (!enabled || !element) return;

    let applied = -1;
    const apply = (progress: number) => {
      if (Math.abs(progress - applied) < 0.0005) return;
      applied = progress;
      const style = element.style;
      style.setProperty("--intro-progress", progress.toFixed(4));
      style.setProperty("--intro-depth", (1 - progress).toFixed(4));
      style.setProperty("--intro-card-progress", clamp01((progress - 0.22) / 0.24).toFixed(4));
      style.setProperty("--intro-copy-progress", clamp01((progress - 0.4) / 0.34).toFixed(4));
    };

    if (reducedMotion) {
      apply(1);
      return;
    }

    let raf = 0;

    const measure = () => {
      raf = 0;
      const rect = element.getBoundingClientRect();
      const viewportHeight = window.innerHeight || 1;
      const start = viewportHeight * 0.72;
      const pinnedTravel = Math.max(rect.height - viewportHeight, 0);
      const end = -Math.max(pinnedTravel * 0.7, viewportHeight * 0.18);
      const raw = clamp01((start - rect.top) / Math.max(start - end, 1));
      apply(easeInOutCubic(raw));
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
  }, [enabled, reducedMotion]);

  return sectionRef;
}

function useAgeValue(birthDate: string) {
  const [age, setAge] = useState(() => calculateAge(birthDate));

  useEffect(() => {
    const update = () => setAge(calculateAge(birthDate));
    update();

    const timer = window.setInterval(update, 60 * 60 * 1000);
    return () => window.clearInterval(timer);
  }, [birthDate]);

  return age;
}

export default function App() {
  const [loadingDismissed, setLoadingDismissed] = useState(false);
  const [headerRef, headerHeight] = useElementHeight<HTMLElement>();
  const viewportHeight = useViewportHeight();
  const shellVisible = useBootReady(BOOT_MIN_MS, BOOT_MAX_MS);
  const shellRef = useScrollReveal(shellVisible);
  const reducedMotion = usePrefersReducedMotion();
  const age = useAgeValue("2009-07-07");
  const profileRef = usePinnedSectionProgress(shellVisible, reducedMotion);
  const emailUser = "11harumune";
  const emailDomainName = "gmail";
  const emailDomainSuffix = "com";
  const emailAddress = `${emailUser}@${emailDomainName}.${emailDomainSuffix}`;
  const discordHandle = "haru0707kun";
  const assetBase = import.meta.env.BASE_URL;
  const avatarImageSrc = `${assetBase}images/avatar-icon.webp`;
  const workBackdropImageSrc = `${assetBase}images/work-night.webp`;

  useEffect(() => {
    if (!shellVisible) {
      setLoadingDismissed(false);
      return;
    }

    const dismissTimer = window.setTimeout(() => {
      setLoadingDismissed(true);
    }, 760);

    return () => window.clearTimeout(dismissTimer);
  }, [shellVisible]);

  const shellStyle = {
    "--header-height": `${headerHeight}px`,
    "--viewport-height": `${viewportHeight}px`,
    "--hero-height": `${Math.max(viewportHeight - headerHeight, 0)}px`,
  } as CSSProperties;

  const profileFacts = [
    {
      label: "BASE",
      value: "Kanagawa / Hokkaido",
    },
    {
      label: "FOCUS",
      value: "DL / LLM",
    },
    {
      label: "AGE",
      value: `${age}`,
    },
    {
      label: "SCHOOL",
      value: "KOSEN 25s",
    },
  ] as const;

  const handleEmailClick = () => {
    window.location.href = `mailto:${emailAddress}?subject=${encodeURIComponent("Portfolio Inquiry")}`;
  };

  return (
    <>
      {!loadingDismissed ? (
        <LoadingScreen
          minMs={BOOT_MIN_MS}
          exiting={shellVisible}
        />
      ) : null}
      <div
        ref={shellRef}
        style={shellStyle}
        className={`${styles.shell} ${
          shellVisible ? styles.shellVisible : styles.shellHidden
        }`}
      >
        <header ref={headerRef} className={styles.header}>
          <a className={styles.logo} href="#top" aria-label="Portfolio home">
            <span className={styles.logoMorph} aria-hidden>
              <span className={styles.logoLetterP}>P</span>
              <span className={styles.logoMiddle}>ort</span>
              <span className={styles.logoLetterF}>
                <span className={styles.logoLetterFUpper}>F</span>
                <span className={styles.logoLetterFLower}>f</span>
              </span>
              <span className={styles.logoTail}>olio</span>
            </span>
          </a>
          <nav className={styles.nav} aria-label="主要ナビゲーション">
            <a href="#about">ABOUT</a>
            <a href="#results">RESULTS</a>
            <a href="#skills">SKILLS</a>
            <a href="#contact">CONTACT</a>
          </nav>
        </header>

        <main id="top">
          <section className={styles.hero} aria-labelledby="hero-title">
            <div className={styles.heroGrid} aria-hidden />
            <div className={styles.heroLayout}>
              <div className={styles.heroCopy}>
                <p className={`${styles.heroEyebrow} mono`}>DL / LLM / KAGGLE</p>
                <h1 id="hero-title" className={styles.heroTitle}>
                  <span className={styles.heroLine}>
                    <span>AI と体験を</span>
                  </span>
                  <span className={styles.heroLine}>
                    <span>静かに</span>
                  </span>
                  <span className={styles.heroLine}>
                    <span>つなぐ</span>
                  </span>
                </h1>
                <p className={styles.heroLead}>
                  高専に通う {age} 歳。DL と LLM を軸に、Kaggle や AI 競技で手を動かしながら、
                  Web の設計と実装まで横断しています。
                </p>
                <div className={styles.heroMeta}>
                  <span className="mono">#KOSEN25S</span>
                  <span className={styles.metaDivider} />
                  <span className="mono">KANAGAWA / HOKKAIDO · JAPAN</span>
                </div>
              </div>
              <div className={styles.heroStage}>
                <AttentionHero active={shellVisible} />
              </div>
            </div>
          </section>

          <section className={styles.statBand} aria-label="主な実績">
            {heroStats.map((stat) => (
              <a
                key={stat.label}
                className={styles.statCell}
                href={stat.href}
                style={{ "--accent": stat.accent } as CSSProperties}
              >
                <span className={`${styles.statLabel} mono`}>{stat.label}</span>
                <p className={styles.statValue}>
                  {stat.value}
                  {"unit" in stat ? <small>{stat.unit}</small> : null}
                </p>
                <p className={styles.statSub}>{stat.sub}</p>
              </a>
            ))}
          </section>

          <section
            id="about"
            ref={profileRef}
            className={styles.profilePhase}
            aria-labelledby="profile-title"
          >
            <div className={styles.profileSticky}>
              <div className={styles.profileBackdrop} aria-hidden>
                <div className={`${styles.profileGlow} ${styles.profileGlowBlue}`} />
                <div className={`${styles.profileGlow} ${styles.profileGlowGold}`} />
                <div className={`${styles.profileGlow} ${styles.profileGlowMint}`} />
              </div>
              <div className={styles.profileStage}>
                <div className={styles.profileVisualZone}>
                  <div className={styles.profileVisualPanel}>
                    <div className={styles.profileVisualPlate}>
                      <span className={`${styles.profileVisualLabel} mono`}>SELF-TAUGHT / KOSEN</span>
                      <p className={styles.profileVisualTitle}>
                        高専に通いながら、DL と LLM を独学で学び続けています。
                      </p>
                    </div>
                    <div className={styles.profileVisualFrame}>
                      <img
                        className={styles.profileVisualImage}
                        src={avatarImageSrc}
                        alt="H. Kobayashi のアイコン"
                        loading="lazy"
                        decoding="async"
                      />
                    </div>
                  </div>
                </div>

                <div className={styles.profileCopy}>
                  <p className={`${styles.profileEyebrow} mono`}>PROFILE</p>
                  <div className={styles.profileIdentityBlock}>
                    <h2 id="profile-title" className={styles.profileNameLarge}>
                      H. Kobayashi
                    </h2>
                    <p className={`${styles.profileHandleLarge} mono`}>
                      @{discordHandle}
                    </p>
                  </div>
                  <p className={styles.profileIntro}>
                    DL や LLM に強い関心があり、コンペティションでの検証から Web の設計・実装まで幅広く取り組んでいます。
                    ご依頼をいただけた際には、責任を持って丁寧に全力で向き合います。
                  </p>
                  <div className={styles.profileMetaGrid}>
                    {profileFacts.map((fact) => (
                      <div key={fact.label} className={styles.profileMetaCard}>
                        <span className={`${styles.profileMetaLabel} mono`}>{fact.label}</span>
                        <p className={styles.profileMetaValue}>{fact.value}</p>
                      </div>
                    ))}
                  </div>
                  <div className={styles.profileLinkRow}>
                    {socialLinks.map((link) => (
                      <a
                        key={link.label}
                        className={`${styles.profileLink} mono`}
                        href={link.href}
                        target="_blank"
                        rel="noreferrer"
                      >
                        {link.label}
                        <Arrow />
                      </a>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </section>

          <section
            id="results"
            className={`${styles.section} ${styles.resultsSection}`}
            aria-labelledby="results-title"
          >
            <div className={styles.resultsLayout}>
              <div
                className={`${styles.sectionHead} ${styles.reveal} ${styles.revealLeft}`}
                data-reveal
              >
                <h2 id="results-title" className={styles.sectionTitle}>
                  RESULTS
                </h2>
                <p className={`${styles.sectionKicker} mono`}>ACHIEVEMENTS</p>
                <p className={styles.sectionLead}>
                  AI 競技とプログラミングコンテストで積み上げてきた結果です。
                </p>
              </div>

              <div className={styles.resultsBody}>
                <ol className={styles.awardList}>
                  {awards.map((award, index) => (
                    <li
                      key={award.title}
                      className={`${styles.awardRow} ${styles.reveal} ${styles.revealUp}`}
                      style={{ "--accent": award.accent } as CSSProperties}
                      data-reveal
                      data-reveal-delay={String(index * 70)}
                    >
                      <span className={`${styles.awardYear} mono`}>{award.year}</span>
                      <div>
                        <h3 className={styles.awardTitle}>{award.title}</h3>
                        <p className={styles.awardDesc}>{award.desc}</p>
                      </div>
                      <span className={styles.awardResult}>{award.result}</span>
                    </li>
                  ))}
                </ol>

                <div
                  id="kaggle"
                  className={`${styles.reveal} ${styles.revealUp}`}
                  data-reveal
                >
                  <div className={styles.subHead}>
                    <p className={`${styles.subHeadLabel} mono`}>KAGGLE MEDALS</p>
                    <a
                      className={`${styles.textLink} mono`}
                      href={KAGGLE_URL}
                      target="_blank"
                      rel="noreferrer"
                    >
                      ALL COMPETITIONS
                      <Arrow />
                    </a>
                  </div>
                  <div className={styles.medalPanel}>
                    <ul className={styles.medalList}>
                      {kaggleMedals.map((entry) => (
                        <li key={entry.name} className={styles.medalRow}>
                          <a
                            className={styles.medalLink}
                            href={entry.href}
                            target="_blank"
                            rel="noreferrer"
                          >
                            <span
                              className={`${styles.medalChip} ${
                                entry.medal === "bronze" ? styles.medalBronze : ""
                              } mono`}
                            >
                              {entry.medal.toUpperCase()}
                            </span>
                            <span className={styles.medalName}>
                              {entry.name}
                              <small>{entry.note}</small>
                            </span>
                            <span className={`${styles.medalRank} mono`}>
                              {numberFormat.format(entry.rank)}
                              <em> / {numberFormat.format(entry.teams)}</em>
                            </span>
                            <span className={`${styles.medalTop} mono`}>
                              {formatTopPercent(entry.rank, entry.teams)}
                            </span>
                          </a>
                        </li>
                      ))}
                    </ul>
                  </div>
                  <p className={styles.medalFoot}>
                    BirdCLEF+ 2026 の解法は{" "}
                    <a href={BIRDCLEF_WRITEUP_URL} target="_blank" rel="noreferrer">
                      Kaggle の Solution Write-up
                    </a>{" "}
                    にまとめています。
                  </p>
                </div>
              </div>
            </div>
          </section>

          <section
            id="skills"
            className={`${styles.section} ${styles.workSection}`}
            aria-labelledby="skills-title"
          >
            <div className={styles.workBackdrop} aria-hidden>
              <img
                className={styles.workBackdropImage}
                src={workBackdropImageSrc}
                alt=""
                loading="lazy"
                decoding="async"
              />
            </div>
            <div className={styles.workContent}>
              <div className={styles.workIntroRow}>
                <div
                  className={`${styles.sectionHead} ${styles.reveal} ${styles.revealLeft}`}
                  data-reveal
                >
                  <h2 id="skills-title" className={styles.sectionTitle}>
                    WHAT
                    <br />I DO
                  </h2>
                  <p className={`${styles.sectionKicker} mono`}>CAPABILITIES</p>
                </div>
                <div className={styles.workCapabilityPanel}>
                  <div className={styles.workCapabilityGrid}>
                    {profileCapabilities.map((capability, index) => (
                      <article
                        key={capability.id}
                        className={`${styles.workCapabilityCard} ${styles.reveal} ${styles.revealUp}`}
                        data-reveal
                        data-reveal-delay={String(80 + index * 80)}
                      >
                        <span className={`${styles.workCapabilityIndex} mono`}>
                          {capability.id}
                        </span>
                        <CapabilityIcon name={capability.icon} />
                        <h3 className={styles.workCapabilityTitle}>{capability.title}</h3>
                        <p className={styles.workCapabilityDesc}>{capability.desc}</p>
                      </article>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </section>

          <footer id="contact" className={styles.footer}>
            <div
              className={`${styles.footerRow} ${styles.reveal} ${styles.revealUp}`}
              data-reveal
            >
              <div className={styles.footerIdentity}>
                <div className={styles.footerAvatarFrame}>
                  <img
                    className={styles.footerAvatar}
                    src={avatarImageSrc}
                    alt=""
                    loading="lazy"
                    decoding="async"
                  />
                </div>
                <div className={styles.footerIdentityMeta}>
                  <p className={styles.footerCta}>CONTACT</p>
                  <p className={styles.footerContactNote}>
                    Discord または X の DM からご連絡いただければ、比較的すぐに対応できます。
                    <span className={`${styles.footerContactHandle} mono`}>
                      {` Discord: ${discordHandle}`}
                    </span>
                  </p>
                </div>
              </div>
              <div className={styles.footerActions}>
                <a
                  className={`${styles.profileLink} mono`}
                  href="https://x.com/haru0707kun"
                  target="_blank"
                  rel="noreferrer"
                >
                  X @{discordHandle}
                  <Arrow />
                </a>
                <button
                  type="button"
                  className={styles.footerMail}
                  onClick={handleEmailClick}
                  aria-label="メールを送る"
                  translate="no"
                  data-nosnippet
                >
                  <span>{emailUser}</span>
                  <span className={`${styles.footerMailHint} mono`} aria-hidden>
                    [at]
                  </span>
                  <span>{emailDomainName}</span>
                  <span className={`${styles.footerMailHint} mono`} aria-hidden>
                    [dot]
                  </span>
                  <span>{emailDomainSuffix}</span>
                </button>
              </div>
            </div>
            <div className={styles.footerBottom}>
              <p className={`${styles.footerCopy} mono`}>
                © {new Date().getFullYear()} HARU0707 · ALL RIGHTS RESERVED
              </p>
              <div className={`${styles.footerBottomLinks} mono`}>
                {socialLinks.map((link) => (
                  <a key={link.label} href={link.href} target="_blank" rel="noreferrer">
                    {link.label.toUpperCase()}
                  </a>
                ))}
              </div>
            </div>
          </footer>
        </main>
      </div>
    </>
  );
}
