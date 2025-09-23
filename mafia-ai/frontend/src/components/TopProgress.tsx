import React from "react";
import { clsx } from "clsx";

export type StepInfo = { id: number; title: string };

export function TopProgress({
  steps, active,
}: { steps: StepInfo[]; active: number }) {
  return (
    <div className="top-progress">
      {steps.map((s, i) => (
        <div key={s.id} className={clsx("tp-item", i === active && "tp-active", i < active && "tp-done")}>
          <div className="tp-dot" />
          <div className="tp-title">{s.title}</div>
          {i < steps.length - 1 && <div className="tp-line" />}
        </div>
      ))}
    </div>
  );
}
