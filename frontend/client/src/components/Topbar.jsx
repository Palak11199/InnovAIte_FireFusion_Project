import { useState, useEffect } from "react";
import { Menu, Search, User, Sun, Moon } from "lucide-react";
import { useSidebarCollapse } from "./SidebarCollapseContext";

export default function Topbar({ title = "Dashboard" }) {
  const { toggle } = useSidebarCollapse();
  const [theme, setTheme] = useState(() => {
    const saved = localStorage.getItem("theme");
    if (saved) return saved;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  });

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prev) => (prev === "dark" ? "light" : "dark"));
  };

  return (
    <header className="topbar">
      <button
        type="button"
        className="topbar-sidebar-toggle"
        onClick={toggle}
        aria-label="Toggle sidebar"
        title="Toggle sidebar"
      >
        <Menu size={20} />
      </button>

      <h2>{title}</h2>

      <select>
        <option>Region: Australia</option>
      </select>

      <select>
        <option>Period: 18 Mar 2026, 14:00 - 20:00</option>
      </select>

      <div className="search">
        <Search size={18} />
        <input placeholder="Search locations, incidents, claims..." />
      </div>

      <span className="sync">Updated 2 min ago</span>

      <button
        type="button"
        className="topbar-sidebar-toggle theme-toggle-btn"
        onClick={toggleTheme}
        aria-label="Toggle dark mode"
        title="Toggle dark mode"
        style={{ marginRight: "4px" }}
      >
        {theme === "dark" ? <Sun size={20} /> : <Moon size={20} />}
      </button>

      <button type="button" className="user-btn">
        <User size={20} />
      </button>
    </header>
  );
}
