// src/components/GesturesDock.tsx
import React from 'react';

export type HandMsg = {
  bbox: [number, number, number, number];
  center: [number, number];
  count: number;
  extended?: boolean[];
  owner_id: number | null;
  label: string;
  fingers: number;
};

export default function GesturesDock({ hands }: { hands: HandMsg[] }) {
  // сортируем слева направо как на столе
  const sorted = [...hands].sort((a, b) => a.center[0] - b.center[0]);
  return (
    <div className="w-full rounded-2xl bg-zinc-900/60 border border-zinc-800 p-3 backdrop-blur">
      <div className="flex items-center gap-2 mb-2">
        <div className="text-zinc-300 font-semibold">Жесты</div>
        <div className="text-xs text-zinc-500">({sorted.length})</div>
      </div>
      <div className="flex flex-wrap gap-2">
        {sorted.map((h, idx) => (
          <div
            key={idx}
            className="px-3 py-2 rounded-xl bg-zinc-800/70 border border-zinc-700 text-zinc-100
                       flex items-center gap-2"
          >
            <span className="text-xs px-2 py-1 rounded-lg bg-indigo-600/30 border border-indigo-500/40">
              {h.owner_id ? `#${h.owner_id}` : '—'}
            </span>
            <span className="font-medium">{h.label}</span>
            <span className="text-xs text-zinc-400">({h.fingers})</span>
          </div>
        ))}
        {sorted.length === 0 && (
          <div className="text-sm text-zinc-500">Жесты не обнаружены</div>
        )}
      </div>
    </div>
  );
}
