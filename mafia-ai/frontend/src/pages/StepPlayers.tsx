import React from "react";
import type { PlayersCount } from "../App";

export function StepPlayers({
  value,
  onChange,
}: {
  value: PlayersCount;
  onChange: (v: PlayersCount) => void;
}) {
  return (
    <div className="card glass players">
      <h2>Количество игроков</h2>
      <div className="players-select">
        <button
          className={`pill ${value === 8 ? "pill-on" : ""}`}
          onClick={() => onChange(8)}
        >
          8 игроков
        </button>
        <button
          className={`pill ${value === 10 ? "pill-on" : ""}`}
          onClick={() => onChange(10)}
        >
          10 игроков
        </button>
      </div>
      <p className="muted">
        Можно изменить позже. Рекомендуется 10 для спортивного формата.
      </p>
    </div>
  );
}
