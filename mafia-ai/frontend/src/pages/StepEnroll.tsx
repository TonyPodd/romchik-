import React from "react";
import type { PlayersCount } from "../App";
import { GlassButton } from "../components/GlassButton";

export function StepEnroll({ players }: { players: PlayersCount }) {
  const slots = Array.from({ length: players }, (_, i) => i + 1);
  return (
    <div className="card glass enroll">
      <h2>Авторизация игроков</h2>
      <p className="muted">
        По одному: посмотрите в камеру и плавно поверните голову (как в iPhone).
        После интеграции логики карточки ниже будут заполняться в реальном
        времени.
      </p>
      <div className="enroll-grid">
        {slots.map((n) => (
          <div key={n} className="enroll-tile">
            <div className="tile-number">#{n}</div>
            <div className="tile-status">Ожидание</div>
            <div className="tile-ghostface" />
          </div>
        ))}
      </div>
      <div className="row">
        <GlassButton onClick={() => alert("Сбросить прогресс (заглушка)")}>
          Сбросить
        </GlassButton>
        <GlassButton onClick={() => alert("Тестовая авторизация (заглушка)")}>
          Псевдо-энролл
        </GlassButton>
      </div>
    </div>
  );
}
