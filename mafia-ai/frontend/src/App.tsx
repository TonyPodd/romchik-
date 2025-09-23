// src/App.tsx
import React, { useMemo, useState, useEffect } from "react";
import { AnimatePresence, motion } from "framer-motion";
import "./index.css";
import { GlassButton } from "@/components/GlassButton";
import { StepWelcome } from "@/pages/StepWelcome";
import { StepConsent } from "@/pages/StepConsent";
import { StepTable } from "@/pages/StepTable";
import { StepRoster } from "@/pages/StepRoster";
import { StepReady } from "@/pages/StepReady";
import { TopProgress } from "@/components/TopProgress";
import ThemeToggle from "@/components/ThemeToggle";

// 👉 импортируем новые API-вызовы
import { startVideo, beginTableCalibration, endTableCalibration } from "@/api";

export type PlayersCount = 8 | 10;

export enum Step {
  Welcome = 0,
  Consent = 1,
  Table   = 2,
  Roster  = 3,
  Ready   = 4,
}

export default function App() {
  const [step, setStep] = useState<Step>(Step.Welcome);
  const [players, setPlayers] = useState<PlayersCount>(8);
  const [consented, setConsented] = useState(false);

  // hover-подсветка для стеклянных кнопок
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      const t = e.target as HTMLElement;
      const btn = t.closest(".glass-btn") as HTMLElement | null;
      if (!btn) return;
      const r = btn.getBoundingClientRect();
      btn.style.setProperty("--mx", `${e.clientX - r.left}px`);
      btn.style.setProperty("--my", `${e.clientY - r.top}px`);
    };
    window.addEventListener("mousemove", handler);
    return () => window.removeEventListener("mousemove", handler);
  }, []);

  // 👉 при входе на шаг «Стол» включаем режим калибровки, при выходе — выключаем
  useEffect(() => {
    let cancelled = false;

    (async () => {
      if (step === Step.Table) {
        // на всякий — гарантируем, что камера поднята
        try { await startVideo({ fps: 30 }); } catch {}
        try { await beginTableCalibration(); } catch {}
      } else {
        try { await endTableCalibration(); } catch {}
      }
      if (cancelled) return;
    })();

    return () => { cancelled = true; };
  }, [step]);

  const steps = useMemo(
    () => [
      { id: Step.Welcome, title: "Приветствие" },
      { id: Step.Consent, title: "Соглашение" },
      { id: Step.Table,   title: "Стол" },
      { id: Step.Roster,  title: "Состав" },
      { id: Step.Ready,   title: "Готово" },
    ],
    []
  );

  const goPrev = () => setStep((s) => Math.max(Step.Welcome, (s - 1) as Step));

  return (
    <div className="app-viewport">
      <div className="bg-grid" />
      <div className="bg-glow bg-glow-1" />
      <div className="bg-glow bg-glow-2" />
      <div className="bg-noise" />

      <header className="topbar">
        <div className="brand">Mafia<span>AI</span></div>
        <div className="spacer" />
        <TopProgress steps={steps} active={step} />
        <div style={{ width: 10 }} />
        <ThemeToggle />
      </header>

      <main className="stage">
        <AnimatePresence mode="wait">
          {step === Step.Welcome && (
            <motion.div key="welcome" initial={{opacity:0,y:24}} animate={{opacity:1,y:0}} exit={{opacity:0,y:-24}} transition={{duration:.5}} className="step-wrap">
              <div className="card glass">
                <div className="card-body hero">
                  <h1 className="hero-title">Mafia<span>AI</span></h1>
                  <p className="lead">ИИ-ведущий для спортивной мафии. Красиво. Быстро. Справедливо.</p>
                  <GlassButton onClick={() => setStep(Step.Consent)}>Начать</GlassButton>
                </div>
              </div>
            </motion.div>
          )}

          {step === Step.Consent && (
            <motion.div key="consent" initial={{opacity:0,y:24,scale:.985}} animate={{opacity:1,y:0,scale:1}} exit={{opacity:0,y:-24,scale:.985}} transition={{type:"spring", stiffness:140, damping:18}} className="step-wrap">
              <div className="card glass">
                <StepConsent consented={consented} onToggle={() => setConsented(v=>!v)} />
                <div className="step-actions">
                  <GlassButton onClick={goPrev} variant="ghost">Назад</GlassButton>
                  <GlassButton onClick={() => setStep(Step.Table)} disabled={!consented}>Продолжить</GlassButton>
                </div>
              </div>
            </motion.div>
          )}

          {step === Step.Table && (
            <motion.div key="table" initial={{opacity:0,scale:.985}} animate={{opacity:1,scale:1}} exit={{opacity:0,scale:.985}} transition={{duration:.5}} className="step-wrap">
              <div className="card glass">
                {/* во время этого шага бэкенд уже в режиме калибровки */}
                <div className="card-body table-step">
                  <StepTable />
                </div>
                <div className="step-actions">
                  <GlassButton onClick={goPrev} variant="ghost">Назад</GlassButton>
                  <GlassButton onClick={() => setStep(Step.Roster)}>Далее</GlassButton>
                </div>
              </div>
            </motion.div>
          )}

          {step === Step.Roster && (
            <motion.div key="roster" initial={{opacity:0,x:24}} animate={{opacity:1,x:0}} exit={{opacity:0,x:-24}} transition={{duration:.5}} className="step-wrap">
              <div className="card glass">
                <StepRoster value={players} onChange={setPlayers} players={players} />
                <div className="step-actions">
                  <GlassButton onClick={goPrev} variant="ghost">Назад</GlassButton>
                  <GlassButton onClick={() => setStep(Step.Ready)}>Завершить</GlassButton>
                </div>
              </div>
            </motion.div>
          )}

          {step === Step.Ready && (
            <motion.div key="ready" initial={{opacity:0,y:30}} animate={{opacity:1,y:0}} exit={{opacity:0}} transition={{duration:.5}} className="step-wrap">
              <div className="card glass">
                <div className="card-body ready">
                  <h2 className="steptitle">Всё готово</h2>
                  <p className="lead">Стол откалиброван, состав определён. Можно начинать матч.</p>
                  <div className="ready-badge">Удачной игры!</div>
                </div>
                <div className="step-actions">
                  <GlassButton onClick={() => setStep(Step.Welcome)} variant="ghost">В начало</GlassButton>
                  <GlassButton onClick={() => alert("Подключим логику позже 🚀")}>Начать игру</GlassButton>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      <footer className="bottombar">
        <div className="badge">pre-alpha UI</div>
      </footer>
    </div>
  );
}
