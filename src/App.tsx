import {
  type CSSProperties,
  useEffect,
  useRef,
  useState,
} from "react";
import { LoadingScreen } from "./LoadingScreen";
import { TransformerHeroAnimation } from "./TransformerHeroAnimation";
import styles from "./App.module.css";

const socialLinks = [
  {
    label: "Kaggle",
    href: "https://www.kaggle.com/haru0707kun",
  },
  {
    label: "GitHub",
    href: "https://github.com/Haru0707kun",
  },
  {
    label: "X",
    href: "https://x.com/Haru0707kun",
  },
] as const;

const profileCapabilities = [
  {
    id: "01",
    title: "AIとUIの統合",
    desc: "AI の機能を UI に自然になじませ、体験として分かりやすく見せる設計ができます。",
  },
  {
    id: "02",
    title: "DLエンジニアリング",
    desc: "DL や LLM を独学で深めながら、検証から実装まで粘り強く進められます。",
  },
  {
    id: "03",
    title: "Webサイト作成",
    desc: "雰囲気を崩さずに、設計・実装・改善・公開まで一貫して対応できます。",
  },
] as const;

const profileAchievements = [
  {
    id: "01",
    title: "日本人工知能オリンピック 銅賞",
    desc: "AI 分野の競技経験を通して、課題理解から実装までを粘り強く積み上げてきました。",
  },
  {
    id: "02",
    title: "U16プログラミングコンテスト 北海道大会 準優勝",
    desc: "ロジックと実装力の両方が問われる場で、北海道大会準優勝の実績があります。",
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

function useBootReady(minMs: number) {
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
      if (raf) cancelAnimationFrame(raf);
      window.removeEventListener("load", onLoad);
    };
  }, [minMs]);

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
  const [progress, setProgress] = useState(reducedMotion ? 1 : 0);

  useEffect(() => {
    if (!enabled) return;

    if (reducedMotion) {
      setProgress(1);
      return;
    }

    let raf = 0;

    const measure = () => {
      raf = 0;
      const element = sectionRef.current;
      if (!element) return;

      const rect = element.getBoundingClientRect();
      const viewportHeight = window.innerHeight || 1;
      const start = viewportHeight * 0.72;
      const end = -Math.min(rect.height * 0.52, viewportHeight * 0.52);
      const raw = clamp01((start - rect.top) / Math.max(start - end, 1));
      setProgress(easeInOutCubic(raw));
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

  return { sectionRef, progress };
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
  const shellVisible = useBootReady(3000);
  const shellRef = useScrollReveal(shellVisible);
  const reducedMotion = usePrefersReducedMotion();
  const age = useAgeValue("2009-07-07");
  const { sectionRef: profileRef, progress: profileProgress } =
    usePinnedSectionProgress(shellVisible, reducedMotion);
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

  const profileStyle = {
    "--intro-progress": profileProgress.toFixed(4),
    "--intro-depth": (1 - profileProgress).toFixed(4),
    "--intro-card-progress": clamp01((profileProgress - 0.22) / 0.24).toFixed(4),
    "--intro-copy-progress": clamp01((profileProgress - 0.4) / 0.34).toFixed(4),
  } as CSSProperties;

  const shellStyle = {
    "--header-height": `${headerHeight}px`,
    "--viewport-height": `${viewportHeight}px`,
    "--hero-height": `${Math.max(viewportHeight - headerHeight, 0)}px`,
  } as CSSProperties;

  const profileFacts = [
    {
      label: "BASE",
      value: "Japan / Hokkaido",
    },
    {
      label: "FOCUS",
      value: "DL / LLM",
    },
    {
      label: "AGE",
      value: `${age}`,
    },
  ] as const;

  const handleEmailClick = () => {
    window.location.href = `mailto:${emailAddress}?subject=${encodeURIComponent("Portfolio Inquiry")}`;
  };

  return (
    <>
      {!loadingDismissed ? (
        <LoadingScreen
          minMs={3000}
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
            <a href="#work">WORK</a>
            <a href="#about">ABOUT</a>
            <a href="#contact">CONTACT</a>
          </nav>
        </header>

        <main id="top">
          <section className={styles.hero} aria-labelledby="hero-title">
            <div className={styles.heroGrid} aria-hidden />
            <div className={styles.heroLayout}>
              <div className={styles.heroCopy}>
                <p className={`${styles.heroEyebrow} mono`}>DL / LLM / WEB</p>
                <h1 id="hero-title" className={styles.heroTitle}>
                  AI と体験を
                  <br />
                  静かに
                  <br />
                  つなぐ
                </h1>
                <p className={styles.heroLead}>
                  DL や LLM への関心を軸に、Web の設計と実装を横断しています。
                  感覚的な心地よさと、長く扱いやすい構造の両立を大切にしています。
                </p>
                <div className={styles.heroMeta}>
                  <span className="mono">HOKKAIDO · JAPAN</span>
                  <span className={styles.metaDivider} />
                  <span className="mono">OPEN 2026</span>
                </div>
              </div>
              <div className={styles.heroStage}>
                <TransformerHeroAnimation active={shellVisible} />
              </div>
            </div>
          </section>

          <section
            id="about"
            ref={profileRef}
            className={styles.profilePhase}
            style={profileStyle}
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
                      <span className={`${styles.profileVisualLabel} mono`}>SELF-TAUGHT / HOKKAIDO</span>
                      <p className={styles.profileVisualTitle}>
                        北海道の高専に通いながら、DL と LLM を独学で学び続けています。
                      </p>
                    </div>
                    <div className={styles.profileVisualFrame}>
                      <img
                        className={styles.profileVisualImage}
                        src={avatarImageSrc}
                        alt="Portrait of Haru"
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
                      H.K.
                    </h2>
                    <p className={`${styles.profileHandleLarge} mono`}>
                      Discord: {discordHandle}
                    </p>
                  </div>
                  <p className={styles.profileIntro}>
                    DL や LLM に強い関心があり、Web の設計から実装まで幅広く取り組んでいます。
                    <br />
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
                      </a>
                    ))}
                  </div>
                  <p className={`${styles.profileSectionLabel} mono`}>ACHIEVEMENTS</p>
                  <div className={styles.profileNoteGrid}>
                    {profileAchievements.map((note) => (
                      <article key={note.id} className={styles.profileNoteCard}>
                        <span className={`${styles.profileNoteIndex} mono`}>{note.id}</span>
                        <h3 className={styles.profileNoteTitle}>{note.title}</h3>
                        <p className={styles.profileNoteDesc}>{note.desc}</p>
                      </article>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </section>

          <section
            id="work"
            className={`${styles.section} ${styles.workSection}`}
            aria-labelledby="work-title"
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
                  <h2 id="work-title" className={styles.sectionTitle}>
                    SELECTED
                    <br />
                    WORK
                  </h2>
                  <p className={`${styles.sectionKicker} mono`}>CAPABILITIES</p>
                </div>
                <div className={styles.workCapabilityPanel}>
                  <p
                    className={`${styles.workCapabilityLabel} mono ${styles.reveal} ${styles.revealRight}`}
                    data-reveal
                    data-reveal-delay="80"
                  >
                    CAN DO
                  </p>
                  <div className={styles.workCapabilityGrid}>
                    {profileCapabilities.map((capability, index) => (
                      <article
                        key={capability.id}
                        className={`${styles.workCapabilityCard} ${styles.reveal} ${styles.revealUp}`}
                        data-reveal
                        data-reveal-delay={String(120 + index * 80)}
                      >
                        <span className={`${styles.workCapabilityIndex} mono`}>
                          {capability.id}
                        </span>
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
                    alt="Haru icon"
                    loading="lazy"
                    decoding="async"
                  />
                </div>
                <div className={styles.footerIdentityMeta}>
                  <p className={styles.footerCta}>CONTACT</p>
                  <p className={styles.footerContactNote}>
                    Discord の DM からご連絡いただければ、比較的すぐに対応できます。
                    <span className={`${styles.footerContactHandle} mono`}>
                      {` Discord: ${discordHandle}`}
                    </span>
                  </p>
                </div>
              </div>
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
            <p
              className={`${styles.footerCopy} mono ${styles.reveal} ${styles.revealUp}`}
              data-reveal
              data-reveal-delay="80"
            >
              © {new Date().getFullYear()} PORTFOLIO · ALL RIGHTS RESERVED
            </p>
          </footer>
        </main>
      </div>
    </>
  );
}
