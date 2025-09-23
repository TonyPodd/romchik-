import React from "react";
import { motion } from "framer-motion";
import { GlassButton } from "../components/GlassButton";

export function StepWelcome({ onStart }: { onStart: () => void }) {
  return (
    <div className="card glass hero">
      <motion.h1
        className="hero-title"
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1, duration: 0.6 }}
      >
        Mafia<span>AI</span>
      </motion.h1>
      <motion.p
        className="hero-sub"
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2, duration: 0.6 }}
      >
        ИИ-ведущий для спортивной мафии. Красиво. Быстро. Справедливо.
      </motion.p>

      <motion.div
        className="hero-privacy"
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.35, duration: 0.6 }}
      >
        Лица и голоса не хранятся в явном виде до и после игры. Используются
        временные эмбеддинги, которые вы можете удалить в любой момент.
      </motion.div>

      <motion.div
        initial={{ opacity: 0, scale: 0.98 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.5, duration: 0.5 }}
      >
        <GlassButton onClick={onStart}>Начать</GlassButton>
      </motion.div>
    </div>
  );
}
