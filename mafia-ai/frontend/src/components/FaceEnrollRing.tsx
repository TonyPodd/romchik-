import React from "react";
import { motion } from "framer-motion";

type Props = {
  /** 0..1 */
  progress: number;
  /** "idle" | "recording" | "done" | "error" */
  state?: "idle" | "recording" | "done" | "error";
  size?: number;           // px
};

export default function FaceEnrollRing({ progress, state="idle", size=220 }: Props){
  const r  = 84;                  // радиус базового круга
  const C  = 2 * Math.PI * r;     // длина окружности
  const p  = Math.max(0, Math.min(1, progress));
  const dash = C * p;

  return (
    <div className="enroll-ring" style={{ width:size, height:size }}>
      <svg className="enroll-svg" viewBox="0 0 220 220" width={size} height={size}>
        {/* мягкие сияния */}
        <defs>
          <linearGradient id="g1" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%"  stopColor="#73c2ff"/>
            <stop offset="100%" stopColor="#8d7bff"/>
          </linearGradient>
          <linearGradient id="g2" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%"  stopColor="#6cffba"/>
            <stop offset="100%" stopColor="#73c2ff"/>
          </linearGradient>
          <filter id="blurSoft" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="10" />
          </filter>
        </defs>

        {/* базовое мутное свечение */}
        <circle cx="110" cy="110" r="90" fill="url(#g1)" opacity=".15" filter="url(#blurSoft)"/>
        <circle cx="110" cy="110" r="70" fill="url(#g2)" opacity=".12" filter="url(#blurSoft)"/>

        {/* тонкая сетка-делений */}
        {[...Array(24)].map((_,i)=> {
          const a = (i/24)*Math.PI*2;
          const x1 = 110 + Math.cos(a)*94;
          const y1 = 110 + Math.sin(a)*94;
          const x2 = 110 + Math.cos(a)*100;
          const y2 = 110 + Math.sin(a)*100;
          return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke="rgba(255,255,255,.18)" strokeWidth={i%6===0?2:1}/>;
        })}

        {/* фон круга */}
        <circle cx="110" cy="110" r={r} stroke="rgba(255,255,255,.16)" strokeWidth="10" fill="none"/>

        {/* прогресс */}
        <motion.circle
          cx="110" cy="110" r={r} fill="none"
          stroke="url(#g1)" strokeWidth="12" strokeLinecap="round"
          strokeDasharray={`${dash} ${C}`}
          transform="rotate(-90 110 110)"
          initial={false}
          animate={{ strokeDasharray: `${dash} ${C}` }}
          transition={{ type:"spring", stiffness:140, damping:20 }}
          style={{ filter: "drop-shadow(0 0 10px rgba(115,194,255,.65))" }}
        />

        {/* бегущая засветка */}
        {state!=="done" && state!=="error" && (
          <motion.g
            animate={{ rotate: 360 }} transition={{ repeat: Infinity, ease:"linear", duration: 2.4 }}
            style={{ transformOrigin:"110px 110px" }}
          >
            <circle cx="110" cy="26" r="6" fill="#fff" opacity=".8"/>
            <circle cx="110" cy="26" r="20" fill="#73c2ff" opacity=".18" filter="url(#blurSoft)"/>
          </motion.g>
        )}
      </svg>

      {/* центр — статус */}
      <div className={`enroll-core ${state}`}>
        {state==="done" ? "✓" : state==="error" ? "!" : `${Math.round(p*100)}%`}
      </div>
    </div>
  );
}
