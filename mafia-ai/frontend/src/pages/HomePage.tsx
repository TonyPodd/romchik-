// src/pages/HomePage.tsx
import React from "react";
import { useNavigate } from "react-router-dom";
import { GlassButton } from "../components/GlassButton";

export const HomePage: React.FC = () => {
  const navigate = useNavigate();

  const features = [
    {
      title: "Автоматическое определение",
      description: "Система автоматически определяет стол и позиции игроков"
    },
    {
      title: "Распознавание лиц",
      description: "Точная идентификация каждого игрока за столом"
    },
    {
      title: "Определение голоса",
      description: "Система распознает кто именно говорит в данный момент"
    },
    {
      title: "Контроль правил",
      description: "Мгновенное выявление нарушений и фолов"
    }
  ];

  return (
    <div className="hero">
      <h1 className="hero-title">
        Mafia<span>AI</span>
      </h1>
      <p className="lead">
        Автоматизированный ведущий для спортивной мафии
      </p>

      <div className="features-grid">
        {features.map((feature, i) => (
          <div key={i} className="feature-card">
            <h3 className="feature-title">{feature.title}</h3>
            <p className="feature-desc">{feature.description}</p>
          </div>
        ))}
      </div>

      <div className="hero-actions">
        <GlassButton
          onClick={() => navigate("/setup/table")}
          size="large"
        >
          Начать настройку
        </GlassButton>
        <GlassButton
          onClick={() => navigate("/setup/voice")}
          variant="ghost"
          size="large"
        >
          Быстрый старт
        </GlassButton>
      </div>
    </div>
  );
};
