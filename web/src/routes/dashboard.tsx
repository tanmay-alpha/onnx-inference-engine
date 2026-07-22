import { createFileRoute, Link } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { LayoutDashboard, TrendingUp, Activity, Cpu, ArrowLeft } from "lucide-react";
import { CrucibleLayout } from "../components/crucible/Layout";
import {
  fetchFraudHistoryFromDB,
  fetchBenchmarksFromDB,
  FraudHistoryRecord,
  BenchmarkRecord,
} from "../lib/api";

interface AnalyticsData {
  inference: {
    period_days: number;
    data: {
      date: string;
      count: number;
      avg_latency_ms: number;
      min_latency_ms: number;
      max_latency_ms: number;
    }[];
  };
  fraud: {
    period_days: number;
    data: {
      date: string;
      total: number;
      fraud_count: number;
      avg_probability: number;
    }[];
  };
  models: {
    id: string;
    name: string;
    usage_count: number;
    inference_count: number;
    avg_latency_ms: number;
    last_used: string | null;
  }[];
}

export const Route = createFileRoute("/dashboard")({
  head: () => ({
    meta: [
      { title: "Analytics Dashboard · Crucible" },
      {
        name: "description",
        content: "Real-time analytics dashboard for Crucible inference platform.",
      },
    ],
  }),
  component: DashboardPage,
});

const fmtNum = (n: number) =>
  new Intl.NumberFormat("en-IN", { maximumFractionDigits: 0 }).format(n);
const fmtPct = (n: number) => `${(n * 100).toFixed(1)}%`;

function StatCard({
  label,
  value,
  sub,
  icon,
}: {
  label: string;
  value: string;
  sub?: string;
  icon: React.ReactNode;
}) {
  return (
    <div className="c-card" style={{ padding: 20 }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          marginBottom: 8,
        }}
      >
        <span style={{ color: "var(--forge)" }}>{icon}</span>
        <span
          className="c-muted"
          style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: ".12em" }}
        >
          {label}
        </span>
      </div>
      <div
        className="mono"
        style={{ fontSize: 28, fontWeight: 600, color: "var(--ink)", letterSpacing: "-.02em" }}
      >
        {value}
      </div>
      {sub && (
        <div className="c-muted" style={{ fontSize: 12, marginTop: 4 }}>
          {sub}
        </div>
      )}
    </div>
  );
}

function DashboardPage() {
  const [analytics, setAnalytics] = useState<AnalyticsData | null>(null);
  const [fraudHistory, setFraudHistory] = useState<FraudHistoryRecord[]>([]);
  const [benchmarks, setBenchmarks] = useState<BenchmarkRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [period, setPeriod] = useState(7);

  useEffect(() => {
    async function load() {
      setLoading(true);
      try {
        const [inf, fraud, mod, benches] = await Promise.all([
          fetch(`/analytics/inference?days=${period}`).then((r) => r.json()),
          fetch(`/analytics/fraud?days=${period}`).then((r) => r.json()),
          fetch("/analytics/models").then((r) => r.json()),
          Promise.all([fetchFraudHistoryFromDB(), fetchBenchmarksFromDB()]),
        ]);
        setAnalytics(inf);
        setFraudHistory(fraud.history || []);
        setBenchmarks(benches[1]);
      } catch {
        // Offline / server not running — show placeholder
        setAnalytics({
          inference: { period_days: period, data: [] },
          fraud: { period_days: period, data: [] },
          models: [],
        });
        setFraudHistory([]);
        setBenchmarks([]);
      }
      setLoading(false);
    }
    load();
  }, [period]);

  const totalInferences = analytics?.inference.data.reduce((s, d) => s + d.count, 0) || 0;
  const totalFraud = fraudHistory.length;
  const fraudCount = fraudHistory.filter((f) => f.probability >= 0.5).length;
  const avgLatency = analytics?.inference.data.length
    ? analytics.inference.data.reduce((s, d) => s + d.avg_latency_ms, 0) /
      analytics.inference.data.length
    : 0;

  return (
    <CrucibleLayout>
      <section className="c-container" style={{ paddingTop: 48, paddingBottom: 56 }}>
        {/* Header */}
        <div
          style={{
            marginBottom: 32,
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            flexWrap: "wrap",
            gap: 16,
          }}
        >
          <div>
            <div className="c-eyebrow">PLATFORM OVERVIEW</div>
            <h1 className="c-h2" style={{ fontSize: 36 }}>
              Analytics Dashboard
            </h1>
            <p className="c-muted" style={{ marginTop: 8 }}>
              Inference volume, fraud detection metrics, and model performance.
            </p>
          </div>
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <label className="c-muted" style={{ fontSize: 12 }}>
              Period:
            </label>
            {[7, 14, 30].map((d) => (
              <button
                key={d}
                className={`c-preset ${period === d ? "active" : ""}`}
                style={{ padding: "6px 14px", width: "auto" }}
                onClick={() => setPeriod(d)}
              >
                {d}d
              </button>
            ))}
            <Link
              to="/"
              className="c-btn c-btn-secondary"
              style={{ padding: "6px 14px", display: "inline-flex", alignItems: "center", gap: 6 }}
            >
              <ArrowLeft size={14} /> Home
            </Link>
          </div>
        </div>

        {loading && (
          <div style={{ textAlign: "center", padding: 40 }}>
            <div className="c-loading-bar" />
            <p className="c-muted" style={{ marginTop: 12 }}>
              Loading analytics...
            </p>
          </div>
        )}

        {!loading && (
          <>
            {/* Stat cards */}
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
                gap: 16,
                marginBottom: 32,
              }}
            >
              <StatCard
                label="Total Inferences"
                value={fmtNum(totalInferences)}
                sub={`Last ${period} days`}
                icon={<Activity size={16} />}
              />
              <StatCard
                label="Fraud Detected"
                value={fmtPct(fraudCount / Math.max(totalFraud, 1))}
                sub={`${fraudCount} of ${totalFraud} checks`}
                icon={<TrendingUp size={16} />}
              />
              <StatCard
                label="Avg Latency"
                value={`${avgLatency.toFixed(1)}ms`}
                sub="Per inference"
                icon={<Cpu size={16} />}
              />
              <StatCard
                label="Active Models"
                value={fmtNum(analytics?.models?.length || 0)}
                sub="Registered"
                icon={<LayoutDashboard size={16} />}
              />
            </div>

            {/* Inference timeline */}
            <div className="c-card" style={{ marginBottom: 24 }}>
              <h3 className="c-h3" style={{ marginBottom: 16 }}>
                Inference Volume
              </h3>
              {analytics?.inference.data.length ? (
                <div
                  style={{
                    display: "flex",
                    alignItems: "flex-end",
                    gap: 4,
                    height: 160,
                    overflowX: "auto",
                  }}
                >
                  {analytics.inference.data.map((d) => (
                    <div
                      key={d.date}
                      style={{
                        flex: "1 1 0",
                        minWidth: 40,
                        display: "flex",
                        flexDirection: "column",
                        alignItems: "center",
                        gap: 4,
                      }}
                    >
                      <div
                        style={{
                          width: "100%",
                          height: `${Math.max(4, (d.count / Math.max(...analytics.inference.data.map((x) => x.count))) * 140)}px`,
                          background: "var(--forge)",
                          borderRadius: 3,
                          opacity: 0.8,
                        }}
                      />
                      <span className="mono" style={{ fontSize: 9, color: "var(--ink-muted)" }}>
                        {new Date(d.date).toLocaleDateString("en", {
                          day: "numeric",
                          month: "short",
                        })}
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="c-muted">
                  No inference data available for this period. Run some inferences to see stats.
                </p>
              )}
            </div>

            {/* Fraud stats */}
            <div className="c-card" style={{ marginBottom: 24 }}>
              <h3 className="c-h3" style={{ marginBottom: 16 }}>
                Fraud Detection Trends
              </h3>
              {fraudHistory.length ? (
                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    gap: 8,
                    maxHeight: 300,
                    overflowY: "auto",
                  }}
                >
                  {fraudHistory.slice(0, 20).map((f) => (
                    <div
                      key={f.id}
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                        padding: "8px 0",
                        borderBottom: "1px solid var(--rule)",
                      }}
                    >
                      <div>
                        <span className="mono" style={{ fontSize: 12, color: "var(--ink-muted)" }}>
                          {new Date(f.created_at).toLocaleString()}
                        </span>
                        <span className="c-muted" style={{ fontSize: 12, marginLeft: 8 }}>
                          {f.tx_type}
                        </span>
                      </div>
                      <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
                        <span className="mono" style={{ fontSize: 13 }}>
                          ₹{fmtNum(f.amount)}
                        </span>
                        <span
                          className="mono"
                          style={{
                            fontSize: 13,
                            color:
                              f.probability >= 0.5
                                ? "var(--risk)"
                                : f.probability >= 0.35
                                  ? "var(--warn)"
                                  : "var(--ok)",
                          }}
                        >
                          {(f.probability * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="c-muted">
                  No fraud detection history available. Run fraud checks to see trends.
                </p>
              )}
            </div>

            {/* Model usage */}
            {analytics?.models?.length ? (
              <div className="c-card">
                <h3 className="c-h3" style={{ marginBottom: 16 }}>
                  Model Usage
                </h3>
                <div style={{ overflowX: "auto" }}>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
                    <thead>
                      <tr style={{ borderBottom: "2px solid var(--rule)" }}>
                        <th
                          style={{
                            textAlign: "left",
                            padding: "8px 12px",
                            fontSize: 11,
                            textTransform: "uppercase",
                            letterSpacing: ".1em",
                            color: "var(--ink-muted)",
                          }}
                        >
                          Model
                        </th>
                        <th
                          style={{
                            textAlign: "right",
                            padding: "8px 12px",
                            fontSize: 11,
                            textTransform: "uppercase",
                            letterSpacing: ".1em",
                            color: "var(--ink-muted)",
                          }}
                        >
                          Usage
                        </th>
                        <th
                          style={{
                            textAlign: "right",
                            padding: "8px 12px",
                            fontSize: 11,
                            textTransform: "uppercase",
                            letterSpacing: ".1em",
                            color: "var(--ink-muted)",
                          }}
                        >
                          Inferences
                        </th>
                        <th
                          style={{
                            textAlign: "right",
                            padding: "8px 12px",
                            fontSize: 11,
                            textTransform: "uppercase",
                            letterSpacing: ".1em",
                            color: "var(--ink-muted)",
                          }}
                        >
                          Avg Latency
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {analytics.models.map((m) => (
                        <tr key={m.id} style={{ borderBottom: "1px solid var(--rule)" }}>
                          <td style={{ padding: "10px 12px" }}>{m.name}</td>
                          <td
                            style={{
                              textAlign: "right",
                              padding: "10px 12px",
                              fontVariantNumeric: "tabular-nums",
                            }}
                          >
                            {fmtNum(m.usage_count)}
                          </td>
                          <td
                            style={{
                              textAlign: "right",
                              padding: "10px 12px",
                              fontVariantNumeric: "tabular-nums",
                            }}
                          >
                            {fmtNum(m.inference_count)}
                          </td>
                          <td
                            style={{
                              textAlign: "right",
                              padding: "10px 12px",
                              fontVariantNumeric: "tabular-nums",
                            }}
                          >
                            {m.avg_latency_ms.toFixed(2)}ms
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : (
              <div className="c-card" style={{ textAlign: "center", padding: 40 }}>
                <p className="c-muted">
                  No model usage data yet. Upload and run models to see analytics.
                </p>
                <Link to="/playground" className="c-btn c-btn-primary" style={{ marginTop: 16 }}>
                  Go to Playground
                </Link>
              </div>
            )}
          </>
        )}
      </section>
    </CrucibleLayout>
  );
}
