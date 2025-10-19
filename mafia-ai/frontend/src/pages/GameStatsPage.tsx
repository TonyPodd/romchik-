// src/pages/GameStatsPage.tsx
import React from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { GlassButton } from "../components/GlassButton";

export const GameStatsPage: React.FC = () => {
  const navigate = useNavigate();

  const mockStats = {
    winner: "Мирные жители",
    duration: "45 минут",
    days: 3,
    eliminated: [
      { name: "Игрок 3", role: "Мафия", day: 1 },
      { name: "Игрок 7", role: "Мирный", day: 2 },
      { name: "Игрок 1", role: "Дон", day: 3 },
    ],
    fouls: [
      { player: "Игрок 5", type: "Речь не в свой ход", count: 2 },
      { player: "Игрок 8", type: "Жест при голосовании", count: 1 },
    ],
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="stats-page"
    >
      <div className="card glass large">
        <div className="card-header">
          <motion.div
            className="winner-badge"
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: "spring", delay: 0.2 }}
          >
            🏆 Победа: {mockStats.winner}
          </motion.div>
        </div>

        <div className="card-body stats-grid">
          {/* Game info */}
          <motion.div
            className="stat-card glass"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            <h3>Информация об игре</h3>
            <div className="stat-item">
              <span className="stat-label">Длительность:</span>
              <span className="stat-value">{mockStats.duration}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">Дней:</span>
              <span className="stat-value">{mockStats.days}</span>
            </div>
          </motion.div>

          {/* Eliminated players */}
          <motion.div
            className="stat-card glass"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
          >
            <h3>Выбывшие игроки</h3>
            {mockStats.eliminated.map((e, i) => (
              <div key={i} className="eliminated-item">
                <span className="player-name">{e.name}</span>
                <span className="player-role">{e.role}</span>
                <span className="day-badge">День {e.day}</span>
              </div>
            ))}
          </motion.div>

          {/* Fouls */}
          <motion.div
            className="stat-card glass"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
          >
            <h3>Фолы</h3>
            {mockStats.fouls.map((f, i) => (
              <div key={i} className="foul-stat">
                <span className="foul-player">{f.player}</span>
                <span className="foul-type">{f.type}</span>
                <span className="foul-count">×{f.count}</span>
              </div>
            ))}
          </motion.div>
        </div>

        <div className="card-footer">
          <GlassButton onClick={() => navigate("/")} variant="ghost">
            На главную
          </GlassButton>
          <GlassButton onClick={() => navigate("/game/setup")}>
            Новая игра
          </GlassButton>
        </div>
      </div>
    </motion.div>
  );
};
