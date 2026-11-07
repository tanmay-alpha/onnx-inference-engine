import { createFileRoute, Link } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { ArrowRight, Lock, Cpu, GitBranch } from "lucide-react";
import { CrucibleLayout } from "../components/crucible/Layout";
import {
  getFraudModelBytes,
  runFraudDetection,
  type FraudDetectionParams,
} from "../lib/crucible-wasm";
import { parseOnnxModel } from "../lib/onnx-meta";

export const Route = createFileRoute("/")({ component: Home });

const inr = new Intl.NumberFormat("en-IN", {
  style: "currency",
  currency: "INR",
  maximumFractionDigits: 0,
});

/** Sample transactions scored live in the visitor's browser on page load. */
const SAMPLES: { kind: string; tx: FraudDetectionParams }[] = [
  {
    kind: "transfer",
    tx: {
      type: "TRANSFER",
      amount: 4850,
      oldBalanceOrig: 42000,
      newBalanceOrig: 37150,
      oldBalanceDest: 12000,
      newBalanceDest: 16850,
    },
  },
  {
    kind: "transfer",
    tx: {
      type: "TRANSFER",
      amount: 201350,
      oldBalanceOrig: 201350,
      newBalanceOrig: 0,
      oldBalanceDest: 0,
      newBalanceDest: 0,
    },
  },
  {
    kind: "cash-out",
    tx: {
      type: "CASH_OUT",
      amount: 154290,
      oldBalanceOrig: 155000,
      newBalanceOrig: 710,
      oldBalanceDest: 5000,
      newBalanceDest: 0,
    },
  },
];

interface LiveRun {
  modelBytes: number;
  nodes: string[];
  wasmBytes: number | null;
  rows: { kind: string; amount: number; probability: number; ms: number }[];
}

/** Runs the real fraud model through the WASM runtime and reports what actually happened. */
function LiveInferenceConsole() {
  const [run, setRun] = useState<LiveRun | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const bytes = await getFraudModelBytes();
      const nodes = parseOnnxModel(bytes).nodes.map((n) => n.opType);
      const rows: LiveRun["rows"] = [];
      for (const s of SAMPLES) {
        const r = await runFraudDetection(s.tx);
        rows.push({
          kind: s.kind,
          amount: s.tx.amount,
          probability: r.probability,
          ms: r.latencyMs,
        });
      }
      // Size of the .wasm the runtime actually downloaded (Resource Timing).
      const wasmEntry = performance
        .getEntriesByType("resource")
        .find((e) => e.name.includes("crucible_wasm_bg.wasm")) as
        PerformanceResourceTiming | undefined;
      const len = wasmEntry?.decodedBodySize ?? 0;
      if (!cancelled) {
        setRun({ modelBytes: bytes.length, nodes, wasmBytes: len > 0 ? len : null, rows });
      }
    })().catch((e: unknown) => {
      if (!cancelled) setError(e instanceof Error ? e.message : String(e));
    });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="c-console" style={{ height: "100%", width: "100%" }}>
      <div className="c-console-bar">
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div className="dots">
            <span className="dot" />
            <span className="dot" />
            <span className="dot on" />
          </div>
          <span className="c-console-title">this tab · crucible-wasm</span>
        </div>
        <span className="c-console-title" style={{ color: "var(--console-accent)" }}>
          {error ? "● error" : run ? "● measured" : "● running"}
        </span>
      </div>
      <div className="c-console-body" style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        <div
          style={{
            fontFamily: "var(--f-mono)",
            fontSize: 12,
            color: "var(--console-mute)",
            lineHeight: 1.7,
          }}
        >
          Scored just now in your browser — values below are live, not recorded.
        </div>
        {error && <div className="c-console-val err">{error}</div>}
        {!error && !run && <div className="c-console-val">loading model + runtime…</div>}
        {run && (
          <>
            <div>
              <div className="c-console-row">
                <span className="c-console-key">model</span>
                <span className="c-console-val s">fraud_detector.onnx · {run.modelBytes} B</span>
              </div>
              <div className="c-console-row">
                <span className="c-console-key">graph</span>
                <span className="c-console-val">{run.nodes.join(" → ")}</span>
              </div>
              <div className="c-console-row">
                <span className="c-console-key">runtime</span>
                <span className="c-console-val">
                  Rust → WebAssembly
                  {run.wasmBytes !== null && ` · ${(run.wasmBytes / 1024).toFixed(1)} KB binary`}
                </span>
              </div>
              <div className="c-console-row">
                <span className="c-console-key">network during scoring</span>
                <span className="c-console-val s">none</span>
              </div>
            </div>
            <div>
              <div className="c-console-title" style={{ marginBottom: 8 }}>
                sample transactions
              </div>
              <div className="c-console-log" style={{ maxHeight: "none" }}>
                {run.rows.map((r, i) => (
                  <div
                    key={i}
                    style={{
                      display: "grid",
                      gridTemplateColumns: "68px 1fr 90px 80px",
                      gap: 8,
                      alignItems: "baseline",
                    }}
                  >
                    <span style={{ color: "var(--console-ink)" }}>{r.kind}</span>
                    <span style={{ color: "var(--console-warn)" }}>{inr.format(r.amount)}</span>
                    <span className={r.probability >= 0.5 ? "warn" : "ok"}>
                      p={(r.probability * 100).toFixed(1)}%
                    </span>
                    <span style={{ color: "var(--console-mute)", textAlign: "right" }}>
                      {r.ms < 0.1 ? "<0.1" : r.ms.toFixed(2)} ms
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
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
            <div className="c-eyebrow">ONNX INFERENCE, FROM SCRATCH</div>
            <h1 className="c-hero-title">
              Fraud checks that run <em>in the browser</em>.
            </h1>
            <p className="c-hero-lede">
              Crucible is a hand-written ONNX inference engine: a C++17 core (20 operators, runs
              ResNet18 and MobileNetV2) behind a FastAPI server, plus a small separate Rust runtime
              compiled to WebAssembly that scores this fraud model inside your tab.
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
            <LiveInferenceConsole />
          </div>
        </div>
      </section>

      {/* PROOF STRIP */}
      <section className="c-container" style={{ paddingTop: 8, paddingBottom: 56 }}>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(4, 1fr)",
            gap: 0,
            border: "1px solid var(--rule)",
            background: "var(--paper)",
          }}
          className="c-proof-strip"
        >
          {[
            { v: "20", k: "ONNX ops · native C++" },
            { v: "5", k: "ONNX ops · WASM runtime" },
            { v: "70", k: "parity cases vs ONNX Runtime" },
            { v: "0 bytes", k: "sent during WASM scoring" },
          ].map((m, i) => (
            <div
              key={m.k}
              style={{
                padding: "22px 24px",
                borderLeft: i === 0 ? "none" : "1px solid var(--rule)",
              }}
            >
              <div
                className="mono"
                style={{
                  fontSize: 24,
                  fontWeight: 500,
                  color: "var(--ink)",
                  letterSpacing: "-.02em",
                }}
              >
                {m.v}
              </div>
              <div
                className="mono"
                style={{
                  fontSize: 10,
                  letterSpacing: ".18em",
                  textTransform: "uppercase",
                  color: "var(--ink-muted)",
                  marginTop: 6,
                }}
              >
                {m.k}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* THREE PRODUCT CARDS */}
      <section className="c-container" style={{ paddingTop: 24, paddingBottom: 56 }}>
        <div
          style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 20 }}
          className="c-hero-split"
        >
          <Link
            to="/fraud"
            className="c-card c-card-hover"
            style={{
              display: "flex",
              flexDirection: "column",
              justifyContent: "space-between",
              minHeight: 210,
            }}
          >
            <div>
              <h3 className="c-h3" style={{ fontSize: 20, marginBottom: 10 }}>
                Fraud Demo
              </h3>
              <p className="c-muted">
                Try Indian transaction examples and run a fraud check fully inside the browser.
              </p>
            </div>
            <div
              className="mono"
              style={{
                marginTop: 24,
                color: "var(--forge)",
                fontSize: 11,
                letterSpacing: ".14em",
                textTransform: "uppercase",
              }}
            >
              Run fraud check →
            </div>
          </Link>

          <Link
            to="/playground"
            className="c-card c-card-hover"
            style={{
              display: "flex",
              flexDirection: "column",
              justifyContent: "space-between",
              minHeight: 210,
            }}
          >
            <div>
              <h3 className="c-h3" style={{ fontSize: 20, marginBottom: 10 }}>
                Playground
              </h3>
              <p className="c-muted">
                Inspect model inputs, graph steps, and browser-side execution behaviour.
              </p>
            </div>
            <div
              className="mono"
              style={{
                marginTop: 24,
                color: "var(--forge)",
                fontSize: 11,
                letterSpacing: ".14em",
                textTransform: "uppercase",
              }}
            >
              Inspect model →
            </div>
          </Link>

          <Link
            to="/benchmark"
            className="c-card c-card-hover"
            style={{
              display: "flex",
              flexDirection: "column",
              justifyContent: "space-between",
              minHeight: 210,
            }}
          >
            <div>
              <h3 className="c-h3" style={{ fontSize: 20, marginBottom: 10 }}>
                Benchmark
              </h3>
              <p className="c-muted">
                Measured latency against ONNX Runtime and PyTorch — including where Crucible is
                slower.
              </p>
            </div>
            <div
              className="mono"
              style={{
                marginTop: 24,
                color: "var(--forge)",
                fontSize: 11,
                letterSpacing: ".14em",
                textTransform: "uppercase",
              }}
            >
              View results →
            </div>
          </Link>
        </div>
      </section>

      {/* PROOF / PRIVACY */}
      <section className="c-container" style={{ paddingTop: 24, paddingBottom: 56 }}>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1.2fr 1fr",
            gap: 56,
            alignItems: "start",
          }}
          className="c-hero-split"
        >
          <div>
            <h2
              className="c-h2"
              style={{
                fontFamily: "var(--f-serif)",
                fontSize: "clamp(28px, 3.4vw, 42px)",
                lineHeight: 1.1,
              }}
            >
              Built so the data stays <em>where it starts</em>.
            </h2>
            <p className="c-muted" style={{ fontSize: 16, marginTop: 18, maxWidth: "52ch" }}>
              Crucible ships the model as static bytes and evaluates it locally in the browser
              sandbox. Only the score is returned to the app. Raw transaction features do not leave
              the device.
            </p>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            {[
              {
                icon: <Cpu size={16} />,
                t: "Local WebAssembly execution",
                d: "Kernels run inside the browser's WASM sandbox.",
              },
              {
                icon: <Lock size={16} />,
                t: "No inference network hop",
                d: "Zero bytes are transmitted for scoring.",
              },
              {
                icon: <GitBranch size={16} />,
                t: "Inspectable ONNX graph",
                d: "Every node and shape is visible in the Playground.",
              },
            ].map((p) => (
              <div
                key={p.t}
                style={{
                  display: "grid",
                  gridTemplateColumns: "36px 1fr",
                  gap: 14,
                  alignItems: "start",
                  paddingBottom: 14,
                  borderBottom: "1px solid var(--rule)",
                }}
              >
                <div
                  className="c-feature-icon-square"
                  style={{
                    width: 30,
                    height: 30,
                    color: "var(--forge)",
                    borderColor: "var(--forge)",
                  }}
                >
                  {p.icon}
                </div>
                <div>
                  <div style={{ fontWeight: 600, color: "var(--ink)", fontSize: 14 }}>{p.t}</div>
                  <div className="c-muted" style={{ fontSize: 13, marginTop: 2 }}>
                    {p.d}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </CrucibleLayout>
  );
}
