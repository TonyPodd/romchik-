import { InputHTMLAttributes } from 'react';

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
}

export function Input({ label, error, style = {}, ...props }: InputProps) {
  return (
    <div style={{ width: '100%' }}>
      {label && (
        <label style={{
          display: 'block',
          marginBottom: '0.5rem',
          fontSize: '0.875rem',
          fontWeight: 500,
          color: '#e2e8f0',
        }}>
          {label}
        </label>
      )}
      <input
        style={{
          width: '100%',
          padding: '0.75rem 1rem',
          background: '#252938',
          border: `1px solid ${error ? '#ef4444' : 'rgba(255,255,255,0.08)'}`,
          borderRadius: '8px',
          color: '#e2e8f0',
          fontSize: '0.9375rem',
          fontFamily: 'inherit',
          transition: 'border 0.15s ease',
          ...style,
        }}
        {...props}
      />
      {error && (
        <div style={{
          marginTop: '0.5rem',
          fontSize: '0.875rem',
          color: '#ef4444',
        }}>
          {error}
        </div>
      )}
    </div>
  );
}
