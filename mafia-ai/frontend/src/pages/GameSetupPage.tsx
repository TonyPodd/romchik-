// src/pages/GameSetupPage.tsx
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { GlassButton } from "../components/GlassButton";

type Role = "mafia" | "don" | "sheriff" | "civilian";

interface PlayerRole {
  id: number;
  name: string;
  role: Role | null;
}

export const GameSetupPage: React.FC = () => {
  const navigate = useNavigate();
  const [players, setPlayers] = useState<PlayerRole[]>(
    Array.from({ length: 10 }, (_, i) => ({
      id: i + 1,
      name: `Игрок ${i + 1}`,
      role: null,
    }))
  );

  const assignRole = (playerId: number, role: Role) => {
    setPlayers((prev) =>
      prev.map((p) => (p.id === playerId ? { ...p, role } : p))
    );
  };

  const autoAssignRoles = () => {
    const roles: Role[] = [
      "don",
      "mafia",
      "mafia",
      "sheriff",
      ...Array(6).fill("civilian"),
    ];

    // Shuffle roles
    for (let i = roles.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [roles[i], roles[j]] = [roles[j], roles[i]];
    }

    setPlayers((prev) =>
      prev.map((p, i) => ({ ...p, role: roles[i] }))
    );
  };

  const allAssigned = players.every((p) => p.role !== null);

  const roleColors = {
    mafia: "#ef4444",
    don: "#991b1b",
    sheriff: "#3b82f6",
    civilian: "#10b981",
  };

  const roleLabels = {
    mafia: "Мафия",
    don: "Дон",
    sheriff: "Шериф",
    civilian: "Мирный",
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="setup-page"
    >
      <div className="card glass large">
        <div className="card-header">
          <h2 className="card-title">Назначение ролей</h2>
          <p className="card-subtitle">Распределите роли между игроками</p>
        </div>

        <div className="card-body">
          {/* Role summary */}
          <div className="role-summary">
            <div className="role-count">
              <span className="role-badge" style={{ background: roleColors.don }}>
                Дон: 1
              </span>
              <span className="role-badge" style={{ background: roleColors.mafia }}>
                Мафия: 2
              </span>
              <span className="role-badge" style={{ background: roleColors.sheriff }}>
                Шериф: 1
              </span>
              <span className="role-badge" style={{ background: roleColors.civilian }}>
                Мирные: 6
              </span>
            </div>
            <GlassButton onClick={autoAssignRoles} variant="ghost">
              🎲 Случайно
            </GlassButton>
          </div>

          {/* Players grid */}
          <div className="players-role-grid">
            {players.map((player) => (
              <motion.div
                key={player.id}
                className="player-role-card glass"
                whileHover={{ scale: 1.02 }}
              >
                <div className="player-header">
                  <span className="player-number">{player.id}</span>
                  <span className="player-name">{player.name}</span>
                </div>
                <div className="role-selector">
                  {(["mafia", "don", "sheriff", "civilian"] as Role[]).map((role) => (
                    <button
                      key={role}
                      className={`role-btn ${player.role === role ? "active" : ""}`}
                      style={{
                        backgroundColor:
                          player.role === role
                            ? roleColors[role]
                            : "transparent",
                        borderColor: roleColors[role],
                      }}
                      onClick={() => assignRole(player.id, role)}
                    >
                      {roleLabels[role]}
                    </button>
                  ))}
                </div>
              </motion.div>
            ))}
          </div>
        </div>

        <div className="card-footer">
          <GlassButton onClick={() => navigate("/setup/voice")} variant="ghost">
            ← Настройка
          </GlassButton>
          <GlassButton
            onClick={() => navigate("/game/live")}
            disabled={!allAssigned}
          >
            Начать игру →
          </GlassButton>
        </div>
      </div>
    </motion.div>
  );
};
