import React, { useEffect, useState } from "react";

type Theme = "dark" | "light";

export default function ThemeToggle(){
  const [theme, setTheme] = useState<Theme>(() => (localStorage.getItem("theme") as Theme) || "dark");

  useEffect(()=>{
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
  },[theme]);

  return (
    <button className="theme-toggle" onClick={()=>setTheme(theme==="dark"?"light":"dark")} title="Сменить тему">
      {theme === "dark" ? "☀️" : "🌙"}
    </button>
  );
}
