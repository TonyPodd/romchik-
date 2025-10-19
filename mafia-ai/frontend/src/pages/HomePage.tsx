// src/pages/HomePage.tsx
import React from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { GlassButton } from "../components/GlassButton";

export const HomePage: React.FC = () => {
  const navigate = useNavigate();

  const features = [
    {
      icon: "🎭",
      title: "Умный ведущий",
      description: "ИИ следит за игрой и автоматически выявляет фолы"
    },
    {
      icon: "📹",
      title: "Распознавание лиц",
      description: "Точная идентификация игроков через камеру"
    },
    {
      icon: "🎙️",
      title: "Анализ речи",
      description: "Определение говорящего и контроль времени"
    },
    {
      icon: "⚡",
      title: "Быстро и справедливо",
      description: "Мгновенная реакция на нарушения правил"
    }
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -24 }}
      transition={{ duration: 0.6 }}
      className="step-wrap"
    >
      <div className="card glass hero-card">
        <div className="card-body hero">
          {/* Hero Section */}
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: 0.2, duration: 0.6 }}
          >
            <h1 className="hero-title">
              Mafia<span>AI</span>
            </h1>
            <p className="lead">
              ИИ-ведущий для спортивной мафии
              <br />
              <span className="text-gradient">Красиво. Быстро. Справедливо.</span>
            </p>
          </motion.div>

          {/* Feature Grid */}
          <motion.div
            className="features-grid"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.4 }}
          >
            {features.map((feature, i) => (
              <motion.div
                key={i}
                className="feature-card glass"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 + i * 0.1 }}
                whileHover={{ scale: 1.05, y: -5 }}
              >
                <div className="feature-icon">{feature.icon}</div>
                <h3 className="feature-title">{feature.title}</h3>
                <p className="feature-desc">{feature.description}</p>
              </motion.div>
            ))}
          </motion.div>

          {/* CTA Buttons */}
          <motion.div
            className="hero-actions"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.9 }}
          >
            <GlassButton
              onClick={() => navigate("/setup/table")}
              size="large"
            >
              🚀 Начать настройку
            </GlassButton>
            <GlassButton
              onClick={() => navigate("/game/setup")}
              variant="ghost"
              size="large"
            >
              ⚡ Быстрый старт
            </GlassButton>
          </motion.div>

          {/* Info Badge */}
          <motion.div
            className="info-badge"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1.1 }}
          >
            <span className="pulse-dot" />
            Для игры требуется камера и 10 игроков
          </motion.div>
        </div>
      </div>
    </motion.div>
  );
};
