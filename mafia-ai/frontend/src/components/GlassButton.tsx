import React from "react";
import { clsx } from "clsx";

type Props = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "ghost";
};

export function GlassButton({ className, variant = "primary", ...rest }: Props) {
  return (
    <button
      className={clsx(
        "glass-btn",
        variant === "ghost" && "glass-btn-ghost",
        className
      )}
      {...rest}
    />
  );
}
