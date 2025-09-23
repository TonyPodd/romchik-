import React from "react";

export function ProgressDots({ total, index }: { total: number; index: number }) {
  return (
    <div className="dots">
      {Array.from({ length: total }).map((_, i) => (
        <span key={i} className={`dot ${i <= index ? "dot-on" : ""}`} />
      ))}
    </div>
  );
}
