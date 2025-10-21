// src/components/Layout.tsx
import React, { useEffect } from "react";
import { Outlet, useNavigate, useLocation } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import ThemeToggle from "./ThemeToggle";

export const Layout: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();

  // Enhanced hover effect for glass buttons with smooth radial gradient
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      const t = e.target as HTMLElement;
      const btn = t.closest(".glass-btn, .nav-link, .glass") as HTMLElement | null;
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
      {/* Enhanced animated background layers */}
      <div className="bg-grid" />
      <div className="bg-glow bg-glow-1" />
      <div className="bg-glow bg-glow-2" />
      <div className="bg-glow bg-glow-3" />
      <div className="bg-noise" />

      {/* Floating particles */}
      <div className="bg-particles">
        {Array.from({ length: 20 }).map((_, i) => (
          <motion.div
            key={i}
            className="particle"
            initial={{
              x: Math.random() * window.innerWidth,
              y: Math.random() * window.innerHeight,
              scale: Math.random() * 0.5 + 0.5,
              opacity: Math.random() * 0.3 + 0.1,
            }}
            animate={{
              x: [
                Math.random() * window.innerWidth,
                Math.random() * window.innerWidth,
              ],
              y: [
                Math.random() * window.innerHeight,
                Math.random() * window.innerHeight,
              ],
            }}
            transition={{
              duration: Math.random() * 20 + 20,
              repeat: Infinity,
              repeatType: "reverse",
              ease: "linear",
            }}
          />
        ))}
      </div>

      {/* Glass header with enhanced blur */}
      <motion.header
        className="topbar glass"
        initial={{ y: -100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ type: "spring", damping: 20, stiffness: 100 }}
      >
        <motion.div
          className="brand"
          onClick={() => navigate("/")}
          style={{ cursor: "pointer" }}
          whileHover={{ scale: 1.08, textShadow: "0 0 20px rgba(115, 194, 255, 0.5)" }}
          whileTap={{ scale: 0.95 }}
          transition={{ type: "spring", damping: 15 }}
        >
          Mafia<span>AI</span>
        </motion.div>
        <div className="spacer" />

        {/* Enhanced navigation with staggered animation */}
        <AnimatePresence mode="wait">
          {!isHomePage && (
            <motion.nav
              className="nav-menu"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ staggerChildren: 0.1 }}
            >
              <NavLink to="/" label="Главная" icon="🏠" />
              <NavLink to="/setup/table" label="Настройка" icon="⚙️" />
              <NavLink to="/game/setup" label="Игра" icon="🎮" />
            </motion.nav>
          )}
        </AnimatePresence>

        <div style={{ width: 16 }} />
        <ThemeToggle />
      </motion.header>

      {/* Main content with proper scroll container */}
      <main className="stage">
        <AnimatePresence mode="wait">
          <motion.div
            key={location.pathname}
            initial={{ opacity: 0, y: 20, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -20, scale: 0.98 }}
            transition={{ duration: 0.3, ease: "easeOut" }}
            style={{ width: "100%", height: "100%" }}
          >
            <Outlet />
          </motion.div>
        </AnimatePresence>
      </main>

      {/* Glass footer with pulse animation */}
      <motion.footer
        className="bottombar glass"
        initial={{ y: 100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ type: "spring", damping: 20, stiffness: 100, delay: 0.2 }}
      >
        <motion.div
          className="badge"
          animate={{ opacity: [0.5, 1, 0.5] }}
          transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
        >
          MafiaAI v0.2 • pre-alpha
        </motion.div>
      </motion.footer>
    </div>
  );
};

// Enhanced navigation link with icon and animations
interface NavLinkProps {
  to: string;
  label: string;
  icon?: string;
}

const NavLink: React.FC<NavLinkProps> = ({ to, label, icon }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const isActive = location.pathname === to || location.pathname.startsWith(to + "/");

  return (
    <motion.button
      className={`nav-link glass ${isActive ? "active" : ""}`}
      onClick={() => navigate(to)}
      whileHover={{
        scale: 1.08,
        backgroundColor: "rgba(255, 255, 255, 0.15)",
        boxShadow: "0 8px 32px rgba(115, 194, 255, 0.3)",
      }}
      whileTap={{ scale: 0.95 }}
      transition={{ type: "spring", damping: 15, stiffness: 300 }}
    >
      {icon && <span className="nav-icon">{icon}</span>}
      <span className="nav-label">{label}</span>
      {isActive && (
        <motion.div
          className="nav-active-indicator"
          layoutId="activeNav"
          transition={{ type: "spring", damping: 20, stiffness: 300 }}
        />
      )}
    </motion.button>
  );
};
