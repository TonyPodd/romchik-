import { ButtonHTMLAttributes, ReactNode } from 'react';

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  children: ReactNode;
  variant?: 'primary' | 'secondary' | 'success' | 'danger';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  fullWidth?: boolean;
}

export function Button({
  children,
  variant = 'primary',
  size = 'md',
  loading = false,
  disabled = false,
  fullWidth = false,
  style = {},
  ...props
}: ButtonProps) {
  const baseStyle: React.CSSProperties = {
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '0.5rem',
    fontWeight: 600,
    border: 'none',
    cursor: disabled || loading ? 'not-allowed' : 'pointer',
    opacity: disabled || loading ? 0.5 : 1,
    transition: 'background 0.15s ease',
    width: fullWidth ? '100%' : 'auto',
    fontFamily: 'inherit',
  };

  const variants = {
    primary: { background: '#4f46e5', color: '#fff' },
    secondary: { background: 'transparent', color: '#e2e8f0', border: '1px solid rgba(255,255,255,0.12)' },
    success: { background: '#10b981', color: '#fff' },
    danger: { background: '#ef4444', color: '#fff' },
  };

  const sizes = {
    sm: { padding: '0.5rem 1rem', fontSize: '0.875rem', borderRadius: '6px' },
    md: { padding: '0.75rem 1.5rem', fontSize: '0.9375rem', borderRadius: '8px' },
    lg: { padding: '1rem 2rem', fontSize: '1rem', borderRadius: '8px' },
  };

  return (
    <button
      style={{ ...baseStyle, ...variants[variant], ...sizes[size], ...style }}
      disabled={disabled || loading}
      {...props}
    >
      {loading && (
        <div style={{
          width: '1rem',
          height: '1rem',
          border: '2px solid rgba(255,255,255,0.3)',
          borderTop: '2px solid white',
          borderRadius: '50%',
          animation: 'spin 1s linear infinite',
        }} />
      )}
      {children}
    </button>
  );
}
