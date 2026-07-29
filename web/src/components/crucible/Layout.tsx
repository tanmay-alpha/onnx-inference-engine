import { Link, useRouterState } from "@tanstack/react-router";
import type { ReactNode } from "react";
import { Github, Linkedin, Menu, X, LogOut, LogIn } from "lucide-react";
import { useState } from "react";
import { logout, getToken } from "../../lib/api";

const GITHUB_URL = "https://github.com/tanmay-alpha/Crucible";
const LINKEDIN_URL = "https://www.linkedin.com/in/tanmaymangal/";

const NAV_ITEMS: { to: string; label: string; auth?: boolean }[] = [
  { to: "/", label: "Home" },
  { to: "/fraud", label: "Fraud Demo" },
  { to: "/playground", label: "Playground" },
  { to: "/models", label: "Models" },
  { to: "/dashboard", label: "Dashboard", auth: true },
  { to: "/api-keys", label: "API Keys", auth: true },
  { to: "/benchmark", label: "Benchmark" },
  { to: "/docs", label: "Docs" },
];

function LogoMark() {
  return (
    <svg className="c-logo-mark" viewBox="0 0 18 18" fill="none" aria-hidden>
      <rect x="0.5" y="0.5" width="5" height="5" stroke="currentColor" />
      <rect x="6.5" y="0.5" width="5" height="5" stroke="currentColor" />
      <rect x="12.5" y="0.5" width="5" height="5" stroke="currentColor" />
      <rect x="0.5" y="6.5" width="5" height="5" stroke="currentColor" />
      <rect x="6.5" y="6.5" width="5" height="5" fill="#C2410C" />
      <rect x="12.5" y="6.5" width="5" height="5" stroke="currentColor" />
      <rect x="0.5" y="12.5" width="5" height="5" stroke="currentColor" />
      <rect x="6.5" y="12.5" width="5" height="5" stroke="currentColor" />
      <rect x="12.5" y="12.5" width="5" height="5" stroke="currentColor" />
    </svg>
  );
}

export function CrucibleLayout({ children }: { children: ReactNode }) {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const [menuOpen, setMenuOpen] = useState(false);
  const [isLoggedIn, setIsLoggedIn] = useState(() => !!localStorage.getItem("crucible_token"));
  const [userEmail, setUserEmail] = useState(() => localStorage.getItem("crucible_user") || "");

  const handleLogout = () => {
    logout();
    setIsLoggedIn(false);
    setUserEmail("");
    setMenuOpen(false);
  };

  const visibleNav = NAV_ITEMS.filter((n) => !n.auth || isLoggedIn);

  return (
    <div className="crucible">
      <header className="c-header">
        <div className="c-header-inner">
          <Link to="/" className="c-logo" aria-label="Crucible home">
            <LogoMark />
            CRUCIBLE
          </Link>
          <nav className="c-nav" aria-label="Primary">
            {visibleNav.map((n) => {
              const active = n.to === "/" ? pathname === "/" : pathname.startsWith(n.to);
              return (
                <Link
                  key={n.to}
                  to={n.to}
                  className={`c-nav-link${active ? " active" : ""}`}
                  aria-current={active ? "page" : undefined}
                >
                  {n.label}
                </Link>
              );
            })}
          </nav>

          <div className="c-nav-actions">
            {isLoggedIn ? (
              <>
                <span className="c-user-badge" title={userEmail}>
                  {userEmail.split("@")[0]}
                </span>
                <button
                  onClick={handleLogout}
                  className="c-nav-link"
                  style={{ display: "inline-flex", alignItems: "center", gap: 4 }}
                  aria-label="Log out"
                >
                  <LogOut size={13} />
                  <span className="c-nav-link-label">Logout</span>
                </button>
              </>
            ) : (
              <Link to="/login" className="c-nav-link c-nav-cta">
                <LogIn size={13} />
                <span className="c-nav-link-label">Login</span>
              </Link>
            )}
            <a href={GITHUB_URL} className="c-nav-source" target="_blank" rel="noreferrer noopener">
              <Github size={13} /> <span className="c-nav-link-label">Source</span>
            </a>
            <button
              className="c-mobile-toggle"
              aria-label="Toggle navigation"
              onClick={() => setMenuOpen(!menuOpen)}
            >
              {menuOpen ? <X size={22} /> : <Menu size={22} />}
            </button>
          </div>
        </div>

        {menuOpen && (
          <nav className="c-mobile-nav" aria-label="Mobile primary">
            {visibleNav.map((n) => {
              const active = n.to === "/" ? pathname === "/" : pathname.startsWith(n.to);
              return (
                <Link
                  key={n.to}
                  to={n.to}
                  className={`c-mobile-nav-link${active ? " active" : ""}`}
                  onClick={() => setMenuOpen(false)}
                >
                  {n.label}
                </Link>
              );
            })}
            {isLoggedIn && (
              <button
                onClick={handleLogout}
                className="c-mobile-nav-link"
                style={{
                  textAlign: "left",
                  background: "none",
                  border: "none",
                  cursor: "pointer",
                  font: "inherit",
                }}
              >
                Logout ({userEmail.split("@")[0]})
              </button>
            )}
            {!isLoggedIn && (
              <Link
                to="/login"
                className="c-mobile-nav-link c-mobile-cta"
                onClick={() => setMenuOpen(false)}
              >
                Login
              </Link>
            )}
          </nav>
        )}
      </header>
      <main>{children}</main>

      <footer className="c-footer">
        <div className="c-footer-inner">
          <div className="c-footer-grid">
            <div className="c-footer-col">
              <Link to="/" className="c-logo" aria-label="Crucible home">
                <LogoMark />
                CRUCIBLE
              </Link>
              <p className="c-footer-desc">
                A privacy-first ONNX inference runtime for the browser. A compact fraud model runs
                locally with WebAssembly — transaction data stays in the tab.
              </p>
            </div>
            <div className="c-footer-col">
              <h4>Project</h4>
              <a href={GITHUB_URL} target="_blank" rel="noreferrer noopener">
                GitHub
              </a>
              <Link to="/docs">Docs</Link>
              <Link to="/benchmark">Benchmark</Link>
              <Link to="/architecture">Architecture</Link>
            </div>
            <div className="c-footer-col">
              <h4>More</h4>
              <Link to="/story">Story</Link>
              <Link to="/roadmap">Roadmap</Link>
              <Link to="/fraud">Privacy note</Link>
            </div>
            <div className="c-footer-col">
              <h4>Author</h4>
              <a href={LINKEDIN_URL} target="_blank" rel="noreferrer noopener">
                <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                  <Linkedin size={13} /> Tanmay Mangal
                </span>
              </a>
              <span
                style={{
                  display: "block",
                  padding: "4px 0",
                  color: "var(--ink-subtle)",
                  fontSize: 12,
                }}
              >
                Built by Tanmay Mangal
              </span>
            </div>
          </div>
          <div className="c-footer-bottom">
            <span>© 2026 Crucible · MIT License</span>
            <div className="c-tech-chips">
              <span className="c-tech-chip">C++17</span>
              <span className="c-tech-chip">Rust</span>
              <span className="c-tech-chip">WebAssembly</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
