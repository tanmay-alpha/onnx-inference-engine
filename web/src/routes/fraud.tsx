import { createFileRoute, Link } from "@tanstack/react-router";
import { useState } from "react";
import { Shield, Lock, AlertTriangle, Inbox } from "lucide-react";
import { CrucibleLayout } from "../components/crucible/Layout";
import { runFraudDetection, type FraudDetectionResult } from "../lib/crucible-wasm";
import { logFraudDetection } from "../lib/api";
import { useIsLoggedIn } from "../lib/use-backend-health";

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
  probability: number;
  latencyMs: number;
  modelBytes: number;
  /** "saved", "not-saved", or the server error message when saving failed. */
  saved: string;
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

function FraudPage() {
  const loggedIn = useIsLoggedIn();
  const [tx, setTx] = useState<Tx>(PRESETS[0].tx);
  const [status, setStatus] = useState<"idle" | "running" | "done" | "error">("idle");
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saveToHistory, setSaveToHistory] = useState(false);

  const update = <K extends keyof Tx>(k: K, v: Tx[K]) => setTx((t) => ({ ...t, [k]: v }));

  const run = async () => {
    const vals = [tx.amount, tx.origBefore, tx.origAfter, tx.destBefore, tx.destAfter];
    const labels = [
      "Amount",
      "Origin balance before",
      "Origin balance after",
      "Dest balance before",
      "Dest balance after",
    ];
    const bad = vals.findIndex((v) => !Number.isFinite(v) || v < 0);
    if (bad >= 0) {
      setStatus("error");
      setError(`${labels[bad]} must be a non-negative number.`);
      return;
    }
    setStatus("running");
    setResult(null);
    setError(null);
    let r: FraudDetectionResult;
    try {
      r = await runFraudDetection({
        type: tx.type,
        amount: tx.amount,
        oldBalanceOrig: tx.origBefore,
        newBalanceOrig: tx.origAfter,
        oldBalanceDest: tx.destBefore,
        newBalanceDest: tx.destAfter,
      });
    } catch (err) {
      setError(`WebAssembly inference failed: ${err instanceof Error ? err.message : String(err)}`);
      setStatus("error");
      return;
    }
    const latency = Number(r.latencyMs.toFixed(3));
    let saved: Result["saved"] = "not-saved";
    // Opt-in only: saving sends this transaction to the Crucible API under your account.
    if (loggedIn && saveToHistory) {
      try {
        await logFraudDetection({
          tx_type: tx.type,
          amount: tx.amount,
          orig_before: tx.origBefore,
          orig_after: tx.origAfter,
          dest_before: tx.destBefore,
          dest_after: tx.destAfter,
          probability: r.probability,
          verdict: verdict(r.probability).label,
          execution_mode: "wasm",
          latency_ms: latency,
        });
        saved = "saved";
      } catch (err) {
        saved = `Could not save to history: ${err instanceof Error ? err.message : String(err)}`;
      }
    }
    setResult({ probability: r.probability, latencyMs: latency, modelBytes: r.modelBytes, saved });
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
            A mock banking form scored by a small ONNX model (logistic regression: MatMul → Add →
            Sigmoid over 7 normalised features) running in Crucible's Rust WebAssembly runtime. The
            model file is fetched once; scoring happens in this tab and sends nothing over the
            network.
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
                  step="1"
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
                    step="1"
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
                    step="1"
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
                    step="1"
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
                    step="1"
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
            {loggedIn && (
              <label
                className="c-muted"
                style={{
                  display: "flex",
                  gap: 8,
                  alignItems: "flex-start",
                  fontSize: 12,
                  marginTop: 12,
                }}
              >
                <input
                  type="checkbox"
                  checked={saveToHistory}
                  onChange={(e) => setSaveToHistory(e.target.checked)}
                  style={{ marginTop: 2 }}
                />
                <span>
                  Save this check to my history. This sends the transaction and score to the
                  Crucible API under your account (shown on the{" "}
                  <Link to="/dashboard" style={{ color: "var(--trace)" }}>
                    dashboard
                  </Link>
                  ).
                </span>
              </label>
            )}
            {status === "running" && (
              <div style={{ marginTop: 14 }}>
                <div className="c-loading-bar" />
                <p className="c-muted" style={{ marginTop: 8, fontSize: 12 }}>
                  Executing the model in WebAssembly…
                </p>
              </div>
            )}
            {status === "error" && error && (
              <div role="alert" style={{ marginTop: 14, color: "var(--risk)", fontSize: 13 }}>
                {error}
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
                  Fill the fields and press <em>Analyse transaction</em>. The ONNX model executes
                  inside this tab.
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
                    <div className="c-metric-label">WASM call time</div>
                    <div className="c-metric-value" style={{ fontSize: 20 }}>
                      {result.latencyMs === 0 ? "<0.1" : result.latencyMs} ms
                    </div>
                  </div>
                  <div className="c-metric" style={{ padding: 14 }}>
                    <div className="c-metric-label">Model bytes</div>
                    <div className="c-metric-value" style={{ fontSize: 20 }}>
                      {result.modelBytes} B
                    </div>
                  </div>
                  <div className="c-metric" style={{ padding: 14 }}>
                    <div className="c-metric-label">Bytes sent to score</div>
                    <div className="c-metric-value" style={{ fontSize: 20, color: "var(--ok)" }}>
                      0
                    </div>
                  </div>
                </div>

                <div className="c-privacy-note" style={{ marginTop: 18 }}>
                  <Lock size={16} style={{ marginTop: 2 }} />
                  <div>
                    {result.saved === "saved"
                      ? "Scored locally. You chose to save it, so this transaction and its score were sent to the Crucible API."
                      : result.saved === "not-saved"
                        ? "Scored locally. No transaction data left the browser tab."
                        : `Scored locally. ${result.saved}`}
                  </div>
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
                This demo shows the alternative topology: the model ships as a static file, is
                evaluated in the browser's WebAssembly sandbox, and the app only needs the resulting
                score. Scoring involves no network round trip.
              </p>
            </div>
          </div>
        </div>
      </section>
    </CrucibleLayout>
  );
}
