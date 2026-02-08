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
  className = '',
  style = {},
  ...props
}: ButtonProps) {
  const fullWidthClass = fullWidth ? 'ui-button--full' : '';

  return (
    <button
      className={`ui-button ui-button--${variant} ui-button--${size} ${fullWidthClass} ${className}`.trim()}
      style={{ ...style }}
      disabled={disabled || loading}
      {...props}
    >
      {loading && (
        <span className="ui-button__spinner" />
      )}
      {children}
    </button>
  );
}
