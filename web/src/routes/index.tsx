import { createFileRoute, Link } from "@tanstack/react-router";
import { ArrowRight, Lock, Cpu, GitBranch } from "lucide-react";
import { CrucibleLayout } from "../components/crucible/Layout";

export const Route = createFileRoute("/")({ component: Home });

const inr = new Intl.NumberFormat("en-IN", {
  style: "currency",
  currency: "INR",
  maximumFractionDigits: 0,
});
const fmt = (n: number) => inr.format(n).replace("₹", "₹");

const RECENT: { id: string; kind: string; amount: number; verdict: string; ms: number; risk: "ok" | "warn" }[] = [
  { id: "001", kind: "transfer", amount: 8151,   verdict: "low risk",      ms: 0.66, risk: "ok" },
  { id: "002", kind: "transfer", amount: 201350, verdict: "elevated risk", ms: 0.61, risk: "warn" },
  { id: "003", kind: "cash-out", amount: 154290, verdict: "elevated risk", ms: 0.83, risk: "warn" },
  { id: "004", kind: "cash-out", amount: 7060,   verdict: "low risk",      ms: 1.32, risk: "ok" },
  { id: "005", kind: "payment",  amount: 5492,   verdict: "low risk",      ms: 1.05, risk: "ok" },
];

function InferenceTerminal() {
  return (
    <div className="c-console" style={{ height: "100%" }}>
      <div className="c-console-bar">
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div className="dots"><span className="dot" /><span className="dot" /><span className="dot on" /></div>
          <span className="c-console-title">local inference session</span>
        </div>
        <span className="c-console-title" style={{ color: "var(--console-accent)" }}>● live</span>
      </div>
      <div className="c-console-body" style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        <div style={{ fontFamily: "var(--f-mono)", fontSize: 12, color: "var(--console-mute)", lineHeight: 1.7 }}>
          <span style={{ color: "var(--console-accent)" }}>$</span>{" "}
          <span style={{ color: "var(--console-ink)" }}>crucible run</span>{" "}
          <span style={{ color: "var(--console-ok)" }}>fraud_mlp_v3.onnx</span>{" "}
          <span style={{ color: "var(--console-mute)" }}>--amount</span>{" "}
          <span style={{ color: "var(--console-warn)" }}>₹51,610</span>{" "}
          <span style={{ color: "var(--console-mute)" }}>--channel</span>{" "}
          <span style={{ color: "var(--console-warn)" }}>transfer</span>
        </div>

        <div>
          <div className="c-console-row"><span className="c-console-key">model loaded</span><span className="c-console-val s">fraud_mlp_v3.onnx</span></div>
          <div className="c-console-row"><span className="c-console-key">runtime</span><span className="c-console-val">wasm-simd128</span></div>
          <div className="c-console-row"><span className="c-console-key">latency</span><span className="c-console-val n">1.18 ms</span></div>
          <div className="c-console-row"><span className="c-console-key">network sent</span><span className="c-console-val s">0 bytes</span></div>
          <div className="c-console-row"><span className="c-console-key">prediction</span><span className="c-console-val warn">elevated risk</span></div>
          <div className="c-console-row"><span className="c-console-key">probability</span><span className="c-console-val warn">82.4%</span></div>
        </div>

        <div>
          <div className="c-console-title" style={{ marginBottom: 8 }}>recent activity</div>
          <div className="c-console-log" style={{ maxHeight: "none" }}>
            {RECENT.map((r) => (
              <div key={r.id} style={{ display: "grid", gridTemplateColumns: "34px 68px 1fr 100px 62px", gap: 8, alignItems: "baseline" }}>
                <span style={{ color: "var(--console-mute)" }}>{r.id}</span>
                <span style={{ color: "var(--console-ink)" }}>{r.kind}</span>
                <span style={{ color: "var(--console-warn)" }}>{fmt(r.amount)}</span>
                <span className={r.risk === "ok" ? "ok" : "warn"}>{r.verdict}</span>
                <span style={{ color: "var(--console-mute)", textAlign: "right" }}>{r.ms.toFixed(2)} ms</span>
              </div>
            ))}
            <div style={{ marginTop: 8, color: "var(--console-accent)" }}>
              $ <span className="c-console-caret" />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function Home() {
  return (
    <CrucibleLayout>
      {/* HERO */}
      <section className="c-container" style={{ paddingTop: 72, paddingBottom: 48 }}>
        <div className="c-hero-split" style={{ gap: 56, alignItems: "center" }}>
          <div>
            <div className="c-eyebrow">PRIVATE BROWSER INFERENCE</div>
            <h1 className="c-hero-title">
              Fraud checks that run <em>in the browser</em>.
            </h1>
            <p className="c-hero-lede">
              Crucible runs a compact ONNX fraud model locally with WebAssembly. Transaction
              data stays in the tab. No inference server, no raw features leaving the device.
            </p>
            <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
              <Link to="/fraud" className="c-btn c-btn-primary c-btn-lg">
                Run fraud check <ArrowRight size={14} />
              </Link>
              <Link to="/playground" className="c-btn c-btn-secondary c-btn-lg">
                Open model playground
              </Link>
            </div>
          </div>
          <div style={{ display: "flex" }}>
            <InferenceTerminal />
          </div>
        </div>
      </section>

      {/* PROOF STRIP */}
      <section className="c-container" style={{ paddingTop: 8, paddingBottom: 56 }}>
        <div
          style={{
            display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 0,
            border: "1px solid var(--rule)", background: "var(--paper)",
          }}
          className="c-proof-strip"
        >
          {[
            { v: "3.1 MB",  k: "runtime" },
            { v: "1.18 ms", k: "sample inference" },
            { v: "0 bytes", k: "sent during inference" },
            { v: "8 nodes", k: "fraud graph" },
          ].map((m, i) => (
            <div key={m.k}
              style={{ padding: "22px 24px", borderLeft: i === 0 ? "none" : "1px solid var(--rule)" }}>
              <div className="mono" style={{ fontSize: 24, fontWeight: 500, color: "var(--ink)", letterSpacing: "-.02em" }}>
                {m.v}
              </div>
              <div className="mono" style={{
                fontSize: 10, letterSpacing: ".18em", textTransform: "uppercase",
                color: "var(--ink-muted)", marginTop: 6,
              }}>
                {m.k}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* THREE PRODUCT CARDS */}
      <section className="c-container" style={{ paddingTop: 24, paddingBottom: 56 }}>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 20 }} className="c-hero-split">
          <Link to="/fraud" className="c-card c-card-hover" style={{ display: "flex", flexDirection: "column", justifyContent: "space-between", minHeight: 210 }}>
            <div>
              <h3 className="c-h3" style={{ fontSize: 20, marginBottom: 10 }}>Fraud Demo</h3>
              <p className="c-muted">
                Try Indian transaction examples and run a fraud check fully inside the browser.
              </p>
            </div>
            <div className="mono" style={{ marginTop: 24, color: "var(--forge)", fontSize: 11, letterSpacing: ".14em", textTransform: "uppercase" }}>
              Run fraud check →
            </div>
          </Link>

          <Link to="/playground" className="c-card c-card-hover" style={{ display: "flex", flexDirection: "column", justifyContent: "space-between", minHeight: 210 }}>
            <div>
              <h3 className="c-h3" style={{ fontSize: 20, marginBottom: 10 }}>Playground</h3>
              <p className="c-muted">
                Inspect model inputs, graph steps, and browser-side execution behaviour.
              </p>
            </div>
            <div className="mono" style={{ marginTop: 24, color: "var(--forge)", fontSize: 11, letterSpacing: ".14em", textTransform: "uppercase" }}>
              Inspect model →
            </div>
          </Link>

          <Link to="/benchmark" className="c-card c-card-hover" style={{ display: "flex", flexDirection: "column", justifyContent: "space-between", minHeight: 210 }}>
            <div>
              <h3 className="c-h3" style={{ fontSize: 20, marginBottom: 10 }}>Benchmark</h3>
              <p className="c-muted">
                Compare runtime size, latency, and cold-start against heavier runtimes.
              </p>
            </div>
            <div className="mono" style={{ marginTop: 24, color: "var(--forge)", fontSize: 11, letterSpacing: ".14em", textTransform: "uppercase" }}>
              View results →
            </div>
          </Link>
        </div>
      </section>

      {/* PROOF / PRIVACY */}
      <section className="c-container" style={{ paddingTop: 24, paddingBottom: 56 }}>
        <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", gap: 56, alignItems: "start" }} className="c-hero-split">
          <div>
            <h2 className="c-h2" style={{ fontFamily: "var(--f-serif)", fontSize: "clamp(28px, 3.4vw, 42px)", lineHeight: 1.1 }}>
              Built so the data stays <em>where it starts</em>.
            </h2>
            <p className="c-muted" style={{ fontSize: 16, marginTop: 18, maxWidth: "52ch" }}>
              Crucible ships the model as static bytes and evaluates it locally in the browser
              sandbox. Only the score is returned to the app. Raw transaction features do not
              leave the device.
            </p>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            {[
              { icon: <Cpu size={16} />,       t: "Local WebAssembly execution",  d: "Kernels run inside the browser's WASM sandbox." },
              { icon: <Lock size={16} />,      t: "No inference network hop",      d: "Zero bytes are transmitted for scoring." },
              { icon: <GitBranch size={16} />, t: "Inspectable ONNX graph",         d: "Every node and shape is visible in the Playground." },
            ].map((p) => (
              <div key={p.t} style={{ display: "grid", gridTemplateColumns: "36px 1fr", gap: 14, alignItems: "start", paddingBottom: 14, borderBottom: "1px solid var(--rule)" }}>
                <div className="c-feature-icon-square" style={{ width: 30, height: 30, color: "var(--forge)", borderColor: "var(--forge)" }}>
                  {p.icon}
                </div>
                <div>
                  <div style={{ fontWeight: 600, color: "var(--ink)", fontSize: 14 }}>{p.t}</div>
                  <div className="c-muted" style={{ fontSize: 13, marginTop: 2 }}>{p.d}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

    </CrucibleLayout>
  );
}
