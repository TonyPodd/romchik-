import { ReactNode, HTMLAttributes } from 'react';

interface GlassCardProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode;
}

export function GlassCard({
  children,
  style = {},
  ...props
}: GlassCardProps) {
  return (
    <div
      style={{
        background: '#1a1d2e',
        border: '1px solid rgba(255,255,255,0.08)',
        borderRadius: '12px',
        ...style,
      }}
      {...props}
    >
      {children}
    </div>
  );
}
