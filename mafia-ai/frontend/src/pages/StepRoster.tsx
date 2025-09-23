import React from "react";
import { motion } from "framer-motion";
import { GlassButton } from "@/components/GlassButton";

export type PlayersCount = 8 | 10;

export function StepRoster({
  value,
  onChange,
  players,
}: {
  value: PlayersCount;
  onChange: (v: PlayersCount) => void;
  players: number;
}) {
  const options: PlayersCount[] = [8, 10];

  return (
    <div className="card-body">
      <header>
        <h2 className="steptitle">Состав и авторизация игроков</h2>
        <p className="lead" style={{marginTop:8}}>
          Выберите количество мест за столом и зарегистрируйте игроков. Номера присваиваются в порядке авторизации.
        </p>
      </header>

      <div className="players" style={{marginTop:8}}>
        <div className="muted" style={{marginBottom:8}}>Количество игроков</div>
        <div className="players-select">
          {options.map((p) => (
            <button
              key={p}
              className={`pill ${value === p ? "pill-on" : ""}`}
              onClick={() => onChange(p)}
            >
              {p}
            </button>
          ))}
        </div>
      </div>

      <div className="enroll">
        <div className="muted" style={{marginTop:16}}>Авторизация</div>
        <motion.div className="enroll-grid" initial={{opacity:0}} animate={{opacity:1}} transition={{duration:.35}} style={{marginTop:8}}>
          {Array.from({ length: players }).map((_, i) => (
            <div key={i} className="enroll-tile">
              <div className="tile-number">Игрок #{i + 1}</div>
              <div className="tile-status">Ожидает регистрацию</div>
              <div className="tile-ghostface" />
              {/* позже: превью лица, статусы, кнопки записи */}
            </div>
          ))}
        </motion.div>
      </div>

      <div className="muted" style={{fontSize:12, marginTop:8}}>
        Примечание: сохраняются только эмбеддинги. Удаляются после матча.
      </div>

      <div style={{ marginTop: 12, display:'flex', gap:8 }}>
        <GlassButton variant="ghost" onClick={() => alert('Подключим камеру на следующем этапе')}>
          Открыть камеру
        </GlassButton>
        <GlassButton onClick={() => alert('Добавим анимацию “поверните голову” позже')}>
          Зарегистрировать всех
        </GlassButton>
      </div>
    </div>
  );
}
