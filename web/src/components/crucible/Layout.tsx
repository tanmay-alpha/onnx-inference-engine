import { Link, useRouterState } from "@tanstack/react-router";
import type { ReactNode } from "react";
import { Github, Linkedin } from "lucide-react";

const GITHUB_URL = "https://github.com/tanmay-alpha/Crucible";
const LINKEDIN_URL = "https://www.linkedin.com/in/tanmaymangal/";

const NAV: { to: string; label: string }[] = [
  { to: "/", label: "Home" },
  { to: "/fraud", label: "Fraud Demo" },
  { to: "/playground", label: "Playground" },
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

  return (
    <div className="crucible">
      <header className="c-header">
        <div className="c-header-inner">
          <Link to="/" className="c-logo" aria-label="Crucible home">
            <LogoMark />
            CRUCIBLE
          </Link>
          <nav className="c-nav" aria-label="Primary">
            {NAV.map((n) => {
              const active = n.to === "/" ? pathname === "/" : pathname.startsWith(n.to);
              return (
                <Link key={n.to} to={n.to} className={`c-nav-link${active ? " active" : ""}`}>
                  {n.label}
                </Link>
              );
            })}
          </nav>
          <a href={GITHUB_URL} className="c-nav-source" target="_blank" rel="noreferrer noopener">
            <Github size={13} /> SOURCE
          </a>
        </div>
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
