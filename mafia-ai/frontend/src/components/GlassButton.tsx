import React from "react";
import { clsx } from "clsx";

type Props = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "ghost";
  size?: "small" | "medium" | "large";
};

export function GlassButton({
  className,
  variant = "primary",
  size = "medium",
  ...rest
}: Props) {
  return (
    <button
      className={clsx(
        "glass-btn",
        variant === "ghost" && "glass-btn-ghost",
        size === "small" && "glass-btn-small",
        size === "large" && "glass-btn-large",
        className
      )}
      {...rest}
    />
  );
}
