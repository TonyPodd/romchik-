// src/pages/SetupWizard.tsx
import React from "react";
import { Outlet, useLocation, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";

export const SetupWizard: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();

  const steps = [
    { path: "/setup/table", label: "Стол", icon: "📹" },
    { path: "/setup/players", label: "Игроки", icon: "👥" },
    { path: "/setup/voice", label: "Голос", icon: "🎙️" },
  ];

  const currentStepIndex = steps.findIndex((s) =>
    location.pathname.startsWith(s.path)
  );

  // Redirect to first step if on /setup
  React.useEffect(() => {
    if (location.pathname === "/setup") {
      navigate("/setup/table", { replace: true });
    }
  }, [location.pathname, navigate]);

  return (
    <div className="setup-wizard">
      {/* Progress indicator */}
      <div className="wizard-progress">
        <div className="progress-steps">
          {steps.map((step, index) => (
            <React.Fragment key={step.path}>
              <motion.div
                className={`progress-step ${
                  index === currentStepIndex ? "active" : ""
                } ${index < currentStepIndex ? "completed" : ""}`}
                whileHover={{ scale: 1.05 }}
                onClick={() => navigate(step.path)}
              >
                <div className="step-icon">{step.icon}</div>
                <div className="step-label">{step.label}</div>
                {index < currentStepIndex && (
                  <div className="step-check">✓</div>
                )}
              </motion.div>
              {index < steps.length - 1 && (
                <div
                  className={`progress-line ${
                    index < currentStepIndex ? "completed" : ""
                  }`}
                />
              )}
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* Step content */}
      <motion.div
        key={location.pathname}
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        exit={{ opacity: 0, x: -20 }}
        transition={{ duration: 0.3 }}
      >
        <Outlet />
      </motion.div>
    </div>
  );
};
