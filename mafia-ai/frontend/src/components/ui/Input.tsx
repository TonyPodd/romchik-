import { InputHTMLAttributes } from 'react';

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
}

export function Input({ label, error, style = {}, ...props }: InputProps) {
  return (
    <div className="ui-input-group">
      {label && (
        <label className="ui-input-label">
          {label}
        </label>
      )}
      <input
        className={`ui-input ${error ? 'ui-input--error' : ''}`.trim()}
        style={{
          ...style,
        }}
        {...props}
      />
      {error && (
        <div className="ui-input-error">
          {error}
        </div>
      )}
    </div>
  );
}
