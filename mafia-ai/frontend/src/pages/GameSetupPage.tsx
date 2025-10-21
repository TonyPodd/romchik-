// src/pages/GameSetupPage.tsx
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
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
  const [isShuffling, setIsShuffling] = useState(false);

  const assignRole = (playerId: number, role: Role) => {
    setPlayers((prev) =>
      prev.map((p) => (p.id === playerId ? { ...p, role } : p))
    );
  };

  const autoAssignRoles = () => {
    setIsShuffling(true);
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

    setTimeout(() => {
      setPlayers((prev) =>
        prev.map((p, i) => ({ ...p, role: roles[i] }))
      );
      setIsShuffling(false);
    }, 1000);
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

  const roleIcons = {
    mafia: "🔪",
    don: "👑",
    sheriff: "🔫",
    civilian: "🏠",
  };

  const roleCount = {
    don: players.filter((p) => p.role === "don").length,
    mafia: players.filter((p) => p.role === "mafia").length,
    sheriff: players.filter((p) => p.role === "sheriff").length,
    civilian: players.filter((p) => p.role === "civilian").length,
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="setup-page"
    >
      <div className="card glass large">
        {/* Enhanced header */}
        <div className="card-header">
          <motion.h2
            className="card-title"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
          >
            Назначение ролей
          </motion.h2>
          <motion.p
            className="card-subtitle"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
          >
            Распределите роли между игроками
          </motion.p>
        </div>

        <div className="card-body">
          {/* Enhanced role summary */}
          <motion.div
            className="role-summary glass"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
          >
            <div className="role-count-grid">
              {(["don", "mafia", "sheriff", "civilian"] as Role[]).map((role, i) => (
                <motion.div
                  key={role}
                  className="role-stat glass"
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.3 + i * 0.1 }}
                  whileHover={{
                    scale: 1.1,
                    backgroundColor: `${roleColors[role]}20`,
                  }}
                  style={{
                    borderColor: roleColors[role],
                  }}
                >
                  <span className="role-icon">{roleIcons[role]}</span>
                  <span className="role-name">{roleLabels[role]}</span>
                  <motion.span
                    className="role-value"
                    style={{ color: roleColors[role] }}
                    animate={
                      roleCount[role] > 0
                        ? {
                            scale: [1, 1.2, 1],
                          }
                        : {}
                    }
                    transition={{ duration: 0.3 }}
                  >
                    {roleCount[role]}/{role === "mafia" ? 2 : role === "civilian" ? 6 : 1}
                  </motion.span>
                </motion.div>
              ))}
            </div>
            <GlassButton
              onClick={autoAssignRoles}
              variant="ghost"
              disabled={isShuffling}
            >
              {isShuffling ? "🔄 Распределение..." : "🎲 Случайное распределение"}
            </GlassButton>
          </motion.div>

          {/* Enhanced players grid */}
          <motion.div
            className="players-role-grid"
            variants={{
              hidden: { opacity: 0 },
              show: {
                opacity: 1,
                transition: {
                  staggerChildren: 0.05,
                },
              },
            }}
            initial="hidden"
            animate="show"
          >
            {players.map((player) => (
              <motion.div
                key={player.id}
                className={`player-role-card glass ${
                  player.role ? `has-role role-${player.role}` : ""
                }`}
                variants={{
                  hidden: { opacity: 0, scale: 0.8 },
                  show: { opacity: 1, scale: 1 },
                }}
                whileHover={{
                  scale: 1.03,
                  boxShadow: player.role
                    ? `0 8px 32px ${roleColors[player.role]}40`
                    : "0 8px 32px rgba(115, 194, 255, 0.3)",
                }}
                animate={
                  isShuffling
                    ? {
                        rotateY: [0, 180, 360],
                        scale: [1, 0.9, 1],
                      }
                    : {}
                }
                transition={
                  isShuffling
                    ? {
                        duration: 1,
                        ease: "easeInOut",
                      }
                    : { type: "spring", damping: 15, stiffness: 300 }
                }
              >
                <div className="player-header">
                  <motion.span
                    className="player-number glass"
                    style={
                      player.role
                        ? {
                            backgroundColor: `${roleColors[player.role]}40`,
                            borderColor: roleColors[player.role],
                          }
                        : {}
                    }
                  >
                    {player.id}
                  </motion.span>
                  <span className="player-name">{player.name}</span>
                  {player.role && (
                    <motion.span
                      className="player-role-icon"
                      initial={{ scale: 0, rotate: -180 }}
                      animate={{ scale: 1, rotate: 0 }}
                      transition={{ type: "spring", damping: 10 }}
                    >
                      {roleIcons[player.role]}
                    </motion.span>
                  )}
                </div>

                {/* Role selector buttons */}
                <div className="role-selector">
                  {(["don", "mafia", "sheriff", "civilian"] as Role[]).map((role) => (
                    <motion.button
                      key={role}
                      className={`role-btn glass ${
                        player.role === role ? "active" : ""
                      }`}
                      style={{
                        backgroundColor:
                          player.role === role
                            ? `${roleColors[role]}90`
                            : "transparent",
                        borderColor: roleColors[role],
                        color: player.role === role ? "#ffffff" : roleColors[role],
                      }}
                      onClick={() => assignRole(player.id, role)}
                      whileHover={{
                        scale: 1.05,
                        backgroundColor: `${roleColors[role]}60`,
                      }}
                      whileTap={{ scale: 0.95 }}
                      transition={{ type: "spring", damping: 15 }}
                    >
                      <span className="btn-icon">{roleIcons[role]}</span>
                      <span className="btn-label">{roleLabels[role]}</span>
                      {player.role === role && (
                        <motion.div
                          className="active-indicator"
                          layoutId={`role-${player.id}`}
                          transition={{ type: "spring", damping: 20, stiffness: 300 }}
                        />
                      )}
                    </motion.button>
                  ))}
                </div>
              </motion.div>
            ))}
          </motion.div>

          {/* All assigned indicator */}
          <AnimatePresence>
            {allAssigned && (
              <motion.div
                className="all-assigned-badge glass"
                initial={{ opacity: 0, y: 20, scale: 0.8 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -20, scale: 0.8 }}
                transition={{ type: "spring", damping: 15 }}
              >
                <motion.div
                  className="check-icon"
                  animate={{ rotate: [0, 360] }}
                  transition={{ duration: 0.6, ease: "easeOut" }}
                >
                  ✓
                </motion.div>
                <span>Все роли назначены</span>
              </motion.div>
            )}
          </AnimatePresence>
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
