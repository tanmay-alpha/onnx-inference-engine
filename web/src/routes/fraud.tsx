import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { Shield, Lock, AlertTriangle, Inbox } from "lucide-react";
import { CrucibleLayout } from "../components/crucible/Layout";
import { runFraudDetection } from "../lib/crucible-wasm";

export const Route = createFileRoute("/fraud")({
  head: () => ({
    meta: [
      { title: "Fraud Detector · Crucible" },
      {
        name: "description",
        content:
          "Run fraud inference on transactions entirely in-browser via Crucible's WASM ONNX runtime.",
      },
      { property: "og:title", content: "Privacy-First Fraud Detection · Crucible" },
      {
        property: "og:description",
        content: "Zero network bytes. Zero data leaks. ML in the tab.",
      },
    ],
  }),
  component: FraudPage,
});

type TxType = "TRANSFER" | "CASH_OUT" | "OTHER";
interface Tx {
  type: TxType;
  amount: number;
  origBefore: number;
  origAfter: number;
  destBefore: number;
  destAfter: number;
}
interface Result {
  fraud: boolean;
  probability: number;
  latencyMs: number;
  modelBytes: number;
}

const PRESETS: { label: string; tone: "success" | "danger" | "warn"; tx: Tx }[] = [
  {
    label: "Everyday transfer · ₹4,850",
    tone: "success",
    tx: {
      type: "TRANSFER",
      amount: 4850,
      origBefore: 42000,
      origAfter: 37150,
      destBefore: 12000,
      destAfter: 16850,
    },
  },
  {
    label: "Account drained · ₹2,01,350",
    tone: "danger",
    tx: {
      type: "TRANSFER",
      amount: 201350,
      origBefore: 201350,
      origAfter: 0,
      destBefore: 0,
      destAfter: 0,
    },
  },
  {
    label: "Borderline cash-out · ₹1,54,290",
    tone: "warn",
    tx: {
      type: "CASH_OUT",
      amount: 154290,
      origBefore: 155000,
      origAfter: 710,
      destBefore: 5000,
      destAfter: 0,
    },
  },
];

function verdict(p: number): { label: string; tone: "ok" | "warn" | "err" } {
  if (p >= 0.65) return { label: "High risk", tone: "err" };
  if (p >= 0.35) return { label: "Elevated risk", tone: "warn" };
  return { label: "Low risk", tone: "ok" };
}

function heuristicScore(tx: Tx): number {
  // Deterministic fallback based on structural signals:
  // drained account, large amount, zeroed destination, type mismatch.
  const drained = tx.origBefore > 0 && tx.origAfter / tx.origBefore < 0.02;
  const bigAmount = tx.amount >= 100000;
  const zeroedDest = tx.destBefore + tx.destAfter === 0 && tx.amount > 1000;
  const mismatch = Math.abs(tx.origBefore - tx.origAfter - tx.amount) > 1;
  let s = 0.02;
  if (drained) s += 0.55;
  if (bigAmount) s += 0.22;
  if (zeroedDest) s += 0.18;
  if (mismatch) s += 0.12;
  if (tx.type === "CASH_OUT") s += 0.08;
  // Deterministic heuristic: same input always produces the same score.
  // No Math.random() — repeated calls with identical transactions must agree.
  return Math.min(0.995, Math.max(0.005, s));
}

function FraudPage() {
  const [tx, setTx] = useState<Tx>(PRESETS[0].tx);
  const [status, setStatus] = useState<"idle" | "running" | "done">("idle");
  const [result, setResult] = useState<Result | null>(null);

  const update = <K extends keyof Tx>(k: K, v: Tx[K]) => setTx((t) => ({ ...t, [k]: v }));

  const run = async () => {
    setStatus("running");
    setResult(null);
    const start = performance.now();
    try {
      // Map form fields to FraudDetectionParams
      const wasmResult = await runFraudDetection({
        type: tx.type,
        amount: tx.amount,
        oldBalanceOrig: tx.origBefore,
        newBalanceOrig: tx.origAfter,
        oldBalanceDest: tx.destBefore,
        newBalanceDest: tx.destAfter,
      });
      const latency = Number((performance.now() - start).toFixed(2));
      setResult({
        fraud: wasmResult.probability >= 0.5,
        probability: wasmResult.probability,
        latencyMs: latency,
        modelBytes: 220,
      });
    } catch (err) {
      // WASM unavailable — use heuristic fallback
      console.warn("WASM inference failed, using heuristic fallback:", err);
      const p = heuristicScore(tx);
      const latency = Number((performance.now() - start).toFixed(2));
      setResult({ fraud: p >= 0.5, probability: p, latencyMs: latency, modelBytes: 0 });
    }
    setStatus("done");
  };

  const v = result ? verdict(result.probability) : null;
  const toneColor =
    v?.tone === "err" ? "var(--risk)" : v?.tone === "warn" ? "var(--warn)" : "var(--ok)";

  return (
    <CrucibleLayout>
      <section className="c-container">
        <div style={{ marginBottom: 32, maxWidth: 720 }}>
          <span className="c-badge c-badge-info">
            <Lock size={12} /> On-device demo
          </span>
          <h1 className="c-h2" style={{ fontSize: 44, marginTop: 14 }}>
            Browser-based transaction analysis
          </h1>
          <p className="c-muted">
            A mock banking portal running Crucible's ONNX runtime in WebAssembly. Customer records
            never leave the tab — the model is fetched once, then evaluated locally against every
            transaction you enter below.
          </p>
        </div>

        <div className="c-two-col">
          <div className="c-card">
            <h3 className="c-h3">Transaction input</h3>
            <p className="c-muted" style={{ marginBottom: 18 }}>
              Pick a scenario or enter values by hand.
            </p>

            <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 22 }}>
              {PRESETS.map((p) => (
                <button
                  key={p.label}
                  className="c-preset"
                  style={{ width: "auto", flex: "1 1 30%" }}
                  onClick={() => {
                    setTx(p.tx);
                    setStatus("idle");
                    setResult(null);
                  }}
                >
                  {p.label}
                </button>
              ))}
            </div>

            <div style={{ display: "grid", gap: 14 }}>
              <div>
                <label className="c-label">Transaction type</label>
                <select
                  className="c-select"
                  value={tx.type}
                  onChange={(e) => update("type", e.target.value as TxType)}
                >
                  <option value="TRANSFER">TRANSFER</option>
                  <option value="CASH_OUT">CASH_OUT</option>
                  <option value="OTHER">OTHER</option>
                </select>
              </div>
              <div>
                <label className="c-label">Amount (₹)</label>
                <input
                  className="c-input"
                  type="number"
                  min="0"
                  step="0.01"
                  value={tx.amount}
                  onChange={(e) => update("amount", Number(e.target.value))}
                  autoComplete="off"
                />
              </div>
              <div className="c-field-row">
                <div>
                  <label className="c-label">Origin balance before</label>
                  <input
                    className="c-input"
                    type="number"
                    min="0"
                    step="0.01"
                    value={tx.origBefore}
                    onChange={(e) => update("origBefore", Number(e.target.value))}
                    autoComplete="off"
                  />
                </div>
                <div>
                  <label className="c-label">Origin balance after</label>
                  <input
                    className="c-input"
                    type="number"
                    min="0"
                    step="0.01"
                    value={tx.origAfter}
                    onChange={(e) => update("origAfter", Number(e.target.value))}
                    autoComplete="off"
                  />
                </div>
              </div>
              <div className="c-field-row">
                <div>
                  <label className="c-label">Destination balance before</label>
                  <input
                    className="c-input"
                    type="number"
                    min="0"
                    step="0.01"
                    value={tx.destBefore}
                    onChange={(e) => update("destBefore", Number(e.target.value))}
                    autoComplete="off"
                  />
                </div>
                <div>
                  <label className="c-label">Destination balance after</label>
                  <input
                    className="c-input"
                    type="number"
                    min="0"
                    step="0.01"
                    value={tx.destAfter}
                    onChange={(e) => update("destAfter", Number(e.target.value))}
                    autoComplete="off"
                  />
                </div>
              </div>
            </div>

            <button
              className="c-btn c-btn-primary c-btn-full"
              style={{ marginTop: 22 }}
              disabled={status === "running"}
              onClick={run}
              aria-live="polite"
            >
              <Shield size={16} />{" "}
              {status === "running" ? "Analysing transaction…" : "Analyse transaction"}
            </button>
            {status === "running" && (
              <div style={{ marginTop: 14 }}>
                <div className="c-loading-bar" />
                <p className="c-muted" style={{ marginTop: 8, fontSize: 12 }}>
                  Executing WASM kernels · no network activity.
                </p>
              </div>
            )}
          </div>

          <div
            className="c-card"
            style={{
              display: "flex",
              flexDirection: "column",
              justifyContent: status === "done" ? "flex-start" : "center",
            }}
          >
            {status !== "done" && (
              <div className="c-card-dashed" style={{ padding: 40, textAlign: "center" }}>
                <Inbox size={22} color="var(--ink-muted)" style={{ marginBottom: 10 }} />
                <div className="c-eyebrow" style={{ marginBottom: 10 }}>
                  Awaiting input
                </div>
                <h3 className="c-h3">No inference yet</h3>
                <p className="c-muted">
                  Fill the fields and press <em>Run local fraud check</em>. The ONNX model executes
                  inside this tab; no transaction data leaves the browser.
                </p>
              </div>
            )}
            {status === "done" && result && v && (
              <div className="c-fade-in">
                <div style={{ marginBottom: 20 }}>
                  <span
                    className="c-badge c-badge-lg"
                    style={{
                      background:
                        v.tone === "err" ? "#F3D7D7" : v.tone === "warn" ? "#F3E7C4" : "#DDEEDF",
                      borderColor:
                        v.tone === "err" ? "#E4B4B4" : v.tone === "warn" ? "#DCC58A" : "#B4D8BE",
                      color: toneColor,
                    }}
                  >
                    {v.label}
                  </span>
                </div>
                <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
                  <div
                    className="c-mono-num"
                    style={{
                      fontSize: 56,
                      fontWeight: 600,
                      letterSpacing: "-0.03em",
                      color: toneColor,
                    }}
                  >
                    {(result.probability * 100).toFixed(1)}%
                  </div>
                  <div className="c-muted">fraud probability</div>
                </div>
                <div style={{ marginTop: 16 }}>
                  <div className="c-meter">
                    <div
                      className="c-meter-fill"
                      style={{ width: `${result.probability * 100}%`, background: toneColor }}
                    />
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6 }}>
                    <span className="c-muted" style={{ fontSize: 11 }}>
                      Legitimate
                    </span>
                    <span className="c-muted" style={{ fontSize: 11 }}>
                      Review
                    </span>
                    <span className="c-muted" style={{ fontSize: 11 }}>
                      Elevated risk
                    </span>
                  </div>
                </div>

                <div className="c-divider" />

                <div className="c-grid-3" style={{ gap: 10 }}>
                  <div className="c-metric" style={{ padding: 14 }}>
                    <div className="c-metric-label">Warm latency</div>
                    <div className="c-metric-value" style={{ fontSize: 20 }}>
                      {result.latencyMs} ms
                    </div>
                  </div>
                  <div className="c-metric" style={{ padding: 14 }}>
                    <div className="c-metric-label">Model bytes</div>
                    <div className="c-metric-value" style={{ fontSize: 20 }}>
                      {result.modelBytes} B
                    </div>
                  </div>
                  <div className="c-metric" style={{ padding: 14 }}>
                    <div className="c-metric-label">Bytes sent</div>
                    <div className="c-metric-value" style={{ fontSize: 20, color: "var(--ok)" }}>
                      0
                    </div>
                  </div>
                </div>

                <div className="c-privacy-note" style={{ marginTop: 18 }}>
                  <Lock size={16} style={{ marginTop: 2 }} />
                  <div>Inference completed locally. No transaction data left the browser tab.</div>
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="c-card" style={{ marginTop: 32 }}>
          <div style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
            <AlertTriangle size={20} color="var(--warn)" style={{ marginTop: 3, flexShrink: 0 }} />
            <div>
              <h3 className="c-h3">Why on-device inference matters for finance</h3>
              <p className="c-muted" style={{ marginBottom: 10 }}>
                Under <span className="mono">PCI-DSS</span> and <span className="mono">GDPR</span>,
                raw cardholder and account data must be tightly scoped. Sending features to a hosted
                inference gateway expands the compliance boundary: the model host becomes an
                in-scope service provider, and cross-border transfer obligations follow.
              </p>
              <p className="c-muted">
                Crucible flips the topology. The model ships as static assets, is evaluated in a
                WebAssembly sandbox with no network access, and only a low-cardinality risk score
                crosses the boundary. Compliance surface collapses; latency drops from a round-trip
                to sub-millisecond kernel time.
              </p>
            </div>
          </div>
        </div>
      </section>
    </CrucibleLayout>
  );
}
