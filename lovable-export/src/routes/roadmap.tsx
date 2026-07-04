import { createFileRoute } from "@tanstack/react-router";
import { CrucibleLayout } from "../components/crucible/Layout";
import { ROADMAP, type RoadmapStatus } from "../data/roadmap";

export const Route = createFileRoute("/roadmap")({
  head: () => ({
    meta: [
      { title: "Roadmap · Crucible" },
      { name: "description", content: "What's shipped, in progress, and planned for Crucible — from the pure-Rust WASM build to biometric matching and OCR." },
      { property: "og:title", content: "Crucible — Roadmap" },
      { property: "og:description", content: "Active development: kernel parity, int8 quantization, biometrics, OCR, WebGPU." },
      { property: "og:url", content: "/roadmap" },
    ],
    links: [{ rel: "canonical", href: "/roadmap" }],
  }),
  component: RoadmapPage,
});

const STATUS_ORDER: RoadmapStatus[] = ["shipped", "in-progress", "planned"];
const STATUS_LABEL: Record<RoadmapStatus, string> = {
  shipped: "Shipped",
  "in-progress": "In progress",
  planned: "Planned",
};
const STATUS_COLOR: Record<RoadmapStatus, string> = {
  shipped: "var(--ok)",
  "in-progress": "var(--warn)",
  planned: "var(--ink-muted)",
};

function RoadmapPage() {
  return (
    <CrucibleLayout>
      <section className="c-container" style={{ maxWidth: 900 }}>
        <span className="c-badge c-badge-info">Roadmap</span>
        <h1 className="c-h2" style={{ fontSize: 42, marginTop: 14 }}>What's done and what's next.</h1>
        <p className="c-muted" style={{ maxWidth: "60ch" }}>
          Crucible is active development, not a closed one-off. Below: everything already in the
          engine, what I'm working on now, and the on-device use cases the runtime is being aimed at.
        </p>

        <div style={{ marginTop: 40, display: "flex", flexDirection: "column", gap: 40 }}>
          {STATUS_ORDER.map((status) => {
            const items = ROADMAP.filter((r) => r.status === status);
            return (
              <div key={status}>
                <div style={{ display: "flex", alignItems: "baseline", gap: 12, marginBottom: 16 }}>
                  <h2 className="c-h2" style={{ fontSize: 22, margin: 0 }}>{STATUS_LABEL[status]}</h2>
                  <span className="mono" style={{ color: STATUS_COLOR[status], fontSize: 11, letterSpacing: ".18em", textTransform: "uppercase" }}>
                    {items.length} item{items.length === 1 ? "" : "s"}
                  </span>
                </div>
                <div style={{ border: "1px solid var(--rule)", background: "var(--paper)" }}>
                  {items.map((r, i) => (
                    <div key={r.title} style={{
                      display: "grid",
                      gridTemplateColumns: "110px 1fr",
                      gap: 20,
                      padding: "16px 20px",
                      borderTop: i === 0 ? "none" : "1px solid var(--rule)",
                      alignItems: "start",
                    }}>
                      <div className="mono" style={{
                        fontSize: 10, letterSpacing: ".16em", textTransform: "uppercase",
                        color: STATUS_COLOR[status], paddingTop: 3,
                      }}>
                        {status === "in-progress" ? "in prog." : status}
                      </div>
                      <div>
                        <div style={{ fontSize: 15, fontWeight: 600, color: "var(--ink)", marginBottom: 4 }}>{r.title}</div>
                        <div className="c-muted">{r.detail}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </section>
    </CrucibleLayout>
  );
}
