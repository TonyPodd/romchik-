import React from "react";
import type { PlayersCount } from "../App";

export function StepReady({ players }: { players: PlayersCount }) {
  return (
    <div className="card glass ready">
      <h2>Всё готово</h2>
      <p className="muted">
        Настроено: {players} игроков. Стол и авторизация завершены.
      </p>
      <div className="ready-badge">Готово к игре</div>
    </div>
  );
}
