import { ReactNode, HTMLAttributes } from 'react';

interface GlassCardProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode;
}

export function GlassCard({
  children,
  className = '',
  style = {},
  ...props
}: GlassCardProps) {
  return (
    <div
      className={`ui-glass-card ${className}`.trim()}
      style={{
        ...style,
      }}
      {...props}
    >
      {children}
    </div>
  );
}
