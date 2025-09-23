import React, { useState } from "react";
import { motion } from "framer-motion";
import ModalGlass from "@/components/ModalGlass";

export function StepConsent({
  consented,
  onToggle,
}: { consented: boolean; onToggle: () => void }) {
  const [openLegal, setOpenLegal] = useState(false);

  return (
    <>
      <div className="card-body no-scroll">
        <div className="grid-consent">
          {/* ЛЕВАЯ КОЛОНКА — коротко и по делу */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.35 }}
          >
            <h2 className="steptitle">Приватность и согласие</h2>
            <p className="lead" style={{ marginTop: 8 }}>
              Вся обработка — локально. <b>Лица и голоса не сохраняются</b> в явном виде.
            </p>

            <ul className="bullets slim" style={{ marginTop: 10 }}>
              <li className="bullet">
                <span className="b-ico b-ok" />
                <div>
                  <div className="b-title">Локальная обработка</div>
                  <div className="b-sub">Данные не отправляются в облако.</div>
                </div>
              </li>
              <li className="bullet">
                <span className="b-ico b-enc" />
                <div>
                  <div className="b-title">Вместо фото — эмбеддинги</div>
                  <div className="b-sub">Числовые векторы для игры, с автo-очисткой.</div>
                </div>
              </li>
            </ul>

            <label className="checkbox" style={{ marginTop: 12 }}>
              <input type="checkbox" checked={consented} onChange={onToggle} />
              <div>
                <div style={{ fontWeight: 800 }}>Согласен(на) с условиями</div>
                <div className="muted" style={{ fontSize: 13, marginTop: 4 }}>
                  <button
                    type="button"
                    className="link-ghost"
                    onClick={() => setOpenLegal(true)}
                  >
                    Читать пользовательское соглашение
                  </button>
                </div>
              </div>
            </label>
          </motion.div>

          {/* ПРАВАЯ КОЛОНКА — щит фиксированной формы */}
          <motion.div
            initial={{ opacity: 0, scale: 0.96 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.45 }}
            className="shield-panel"
          >
            <div className="shield-wrap">
              <svg
                className="shield-svg"
                viewBox="0 0 240 280"
                preserveAspectRatio="xMidYMid meet"
                aria-hidden
              >
                <defs>
                  <linearGradient id="g1" x1="0" y1="0" x2="1" y2="1">
                    <stop offset="0%" stopColor="rgba(115,194,255,1)" />
                    <stop offset="100%" stopColor="rgba(141,123,255,1)" />
                  </linearGradient>
                </defs>
                <path
                  d="M120 18 C150 40 180 44 210 44 C208 160 180 210 120 258 C60 210 32 160 30 44 C60 44 90 40 120 18 Z"
                  fill="url(#g1)"
                  opacity="0.25"
                />
                <path
                  d="M120 35 C145 52 170 55 195 55 C193 152 170 195 120 237 C70 195 47 152 45 55 C70 55 95 52 120 35 Z"
                  stroke="url(#g1)"
                  strokeWidth="2.4"
                  fill="none"
                />
                <circle cx="120" cy="140" r="44" fill="none" stroke="url(#g1)" strokeWidth="2" />
                <path d="M120 110 a30 30 0 1 0 0.1 0" fill="none" stroke="url(#g1)" strokeWidth="2" />
              </svg>
              <div className="shield-glow s1" />
              <div className="shield-glow s2" />
            </div>
          </motion.div>
        </div>
      </div>

      {/* Модалка с полным текстом соглашения */}
      <ModalGlass open={openLegal} title="Пользовательское соглашение" onClose={() => setOpenLegal(false)}>
        <div className="legal-scroll" style={{ maxHeight: 360 }}>
          <p> Если вы умрете - вы виноваты сами</p>
          
        </div>
      </ModalGlass>
    </>
  );

  
}
