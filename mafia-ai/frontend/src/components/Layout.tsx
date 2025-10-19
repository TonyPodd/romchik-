// src/components/Layout.tsx
import React, { useEffect } from "react";
import { Outlet, useNavigate, useLocation } from "react-router-dom";
import { motion } from "framer-motion";
import ThemeToggle from "./ThemeToggle";

export const Layout: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();

  // hover-подсветка для стеклянных кнопок
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      const t = e.target as HTMLElement;
      const btn = t.closest(".glass-btn") as HTMLElement | null;
      if (!btn) return;
      const r = btn.getBoundingClientRect();
      btn.style.setProperty("--mx", `${e.clientX - r.left}px`);
      btn.style.setProperty("--my", `${e.clientY - r.top}px`);
    };
    window.addEventListener("mousemove", handler);
    return () => window.removeEventListener("mousemove", handler);
  }, []);

  const isHomePage = location.pathname === "/";

  return (
    <div className="app-viewport">
      {/* Animated background */}
      <div className="bg-grid" />
      <div className="bg-glow bg-glow-1" />
      <div className="bg-glow bg-glow-2" />
      <div className="bg-noise" />

      {/* Header */}
      <header className="topbar">
        <motion.div
          className="brand"
          onClick={() => navigate("/")}
          style={{ cursor: "pointer" }}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          Mafia<span>AI</span>
        </motion.div>
        <div className="spacer" />

        {/* Navigation menu - только если не на главной */}
        {!isHomePage && (
          <nav className="nav-menu">
            <NavLink to="/" label="Главная" />
            <NavLink to="/setup/table" label="Настройка" />
            <NavLink to="/game/setup" label="Игра" />
          </nav>
        )}

        <div style={{ width: 16 }} />
        <ThemeToggle />
      </header>

      {/* Main content */}
      <main className="stage">
        <Outlet />
      </main>

      {/* Footer */}
      <footer className="bottombar">
        <div className="badge">MafiaAI v0.2 • pre-alpha</div>
      </footer>
    </div>
  );
};

// Helper component for navigation links
interface NavLinkProps {
  to: string;
  label: string;
}

const NavLink: React.FC<NavLinkProps> = ({ to, label }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const isActive = location.pathname === to || location.pathname.startsWith(to + "/");

  return (
    <motion.button
      className={`nav-link ${isActive ? "active" : ""}`}
      onClick={() => navigate(to)}
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
    >
      {label}
    </motion.button>
  );
};
