import { createFileRoute, Link, redirect } from "@tanstack/react-router";
import { useEffect, useState, useMemo, useCallback } from "react";
import {
  LayoutDashboard,
  TrendingUp,
  Activity,
  Cpu,
  ArrowLeft,
  RefreshCw,
  BarChart3,
  Shield,
  Zap,
} from "lucide-react";
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { CrucibleLayout } from "../components/crucible/Layout";
import {
  fetchFraudHistoryFromDB,
  fetchBenchmarksFromDB,
  getFraudHistory,
  fetchModels,
} from "../lib/api";
import type { FraudHistoryRecord, BenchmarkRecord, RegisteredModel } from "../lib/api";
import { Button } from "../components/ui/button";

const COLORS = ["#c2410c", "#152A66", "#166534", "#b45309", "#7c3aed", "#0891b2"];

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
  beforeLoad: () => {
    if (!localStorage.getItem("crucible_token")) {
      throw redirect({ to: "/login" });
    }
  },
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
  const [loading, setLoading] = useState(true);
  const [period, setPeriod] = useState(7);
  const [refreshing, setRefreshing] = useState(false);
  const [models, setModels] = useState<RegisteredModel[]>([]);

  const loadData = useCallback(async (days: number) => {
    setLoading(true);
    try {
      const [inf, fraud, mods] = await Promise.all([
        fetch(`/analytics/inference?days=${days}`)
          .then((r) => r.json())
          .catch(() => null),
        fetch(`/analytics/fraud?days=${days}`)
          .then((r) => r.json())
          .catch(() => null),
        fetchModels(),
      ]);

      if (inf) setAnalytics(inf);
      else
        setAnalytics({
          inference: { period_days: days, data: [] },
          fraud: { period_days: days, data: [] },
          models: [],
        });

      setFraudHistory(fraud?.history || []);
      setModels(mods);
    } catch {
      setAnalytics({
        inference: { period_days: days, data: [] },
        fraud: { period_days: days, data: [] },
        models: [],
      });
      setFraudHistory([]);
      setModels([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData(period);
  }, [period, loadData]);

  const handleRefresh = async () => {
    setRefreshing(true);
    await loadData(period);
    setRefreshing(false);
  };

  const totalInferences = analytics?.inference.data.reduce((s, d) => s + d.count, 0) || 0;
  const totalFraud = fraudHistory.length;
  const fraudCount = fraudHistory.filter((f) => f.probability >= 0.5).length;
  const avgLatency = analytics?.inference.data.length
    ? analytics.inference.data.reduce((s, d) => s + d.avg_latency_ms, 0) /
      analytics.inference.data.length
    : 0;

  // Build latency distribution bins from fraud history + analytics
  const latencyBins = useMemo(() => {
    const bins = [
      { range: "0-1ms", count: 0 },
      { range: "1-5ms", count: 0 },
      { range: "5-10ms", count: 0 },
      { range: "10-50ms", count: 0 },
      { range: "50ms+", count: 0 },
    ];

    const latencies = fraudHistory.map((f) => f.latency_ms);
    analytics?.inference.data.forEach((d) => {
      latencies.push(d.avg_latency_ms);
    });

    latencies.forEach((ms) => {
      if (ms < 1) bins[0].count++;
      else if (ms < 5) bins[1].count++;
      else if (ms < 10) bins[2].count++;
      else if (ms < 50) bins[3].count++;
      else bins[4].count++;
    });

    return bins.filter((b) => b.count > 0);
  }, [fraudHistory, analytics]);

  // Model usage pie data
  const modelPieData = useMemo(() => {
    if (!models.length) return [];
    return models
      .map((m) => ({
        name: m.name.replace(".onnx", ""),
        value: m.usage_count,
      }))
      .filter((m) => (m as { name: string; value: number }).value > 0);
  }, [models]);

  const inferenceChartData = useMemo(() => {
    return (analytics?.inference.data || []).map((d) => ({
      ...d,
      dateLabel: new Date(d.date).toLocaleDateString("en", {
        month: "short",
        day: "numeric",
      }),
    }));
  }, [analytics]);

  const fraudChartData = useMemo(() => {
    return (analytics?.fraud.data || []).map((d) => ({
      ...d,
      dateLabel: new Date(d.date).toLocaleDateString("en", {
        month: "short",
        day: "numeric",
      }),
    }));
  }, [analytics]);

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
          <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
            <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
              <span className="c-muted" style={{ fontSize: 12 }}>
                Period:
              </span>
              {[7, 14, 30].map((d) => (
                <button
                  key={d}
                  className={`c-preset ${period === d ? "active" : ""}`}
                  style={{
                    padding: "6px 14px",
                    width: "auto",
                    background: period === d ? "var(--forge-tint)" : undefined,
                    borderColor: period === d ? "var(--trace)" : undefined,
                  }}
                  onClick={() => setPeriod(d)}
                >
                  {d}d
                </button>
              ))}
            </div>
            <Button
              variant="secondary"
              onClick={handleRefresh}
              disabled={refreshing}
              style={{ padding: "6px 14px" }}
            >
              <RefreshCw size={14} className={refreshing ? "c-spin" : ""} />
              Refresh
            </Button>
            <Link
              to="/"
              className="c-btn c-btn-secondary"
              style={{ padding: "6px 14px", display: "inline-flex", alignItems: "center", gap: 6 }}
            >
              <ArrowLeft size={14} /> Home
            </Link>
          </div>
        </div>

        {loading ? (
          <div style={{ textAlign: "center", padding: 40 }}>
            <div className="c-loading-bar" />
            <p className="c-muted" style={{ marginTop: 12 }}>
              Loading analytics...
            </p>
          </div>
        ) : (
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
                icon={<Zap size={16} />}
              />
              <StatCard
                label="Active Models"
                value={fmtNum(models.length)}
                sub="Registered"
                icon={<Cpu size={16} />}
              />
            </div>

            {/* Row 1: Inference volume + Fraud trend */}
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(380px, 1fr))",
                gap: 20,
                marginBottom: 24,
              }}
            >
              {/* Inference volume line chart */}
              <div className="c-card">
                <h3
                  className="c-h3"
                  style={{ marginBottom: 16, display: "flex", alignItems: "center", gap: 8 }}
                >
                  <BarChart3 size={16} style={{ color: "var(--forge)" }} />
                  Inference Volume
                </h3>
                {inferenceChartData.length > 0 ? (
                  <div style={{ width: "100%", height: 260 }}>
                    <ResponsiveContainer>
                      <AreaChart
                        data={inferenceChartData}
                        margin={{ top: 5, right: 5, left: -10, bottom: 5 }}
                      >
                        <defs>
                          <linearGradient id="inferenceGrad" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#c2410c" stopOpacity={0.15} />
                            <stop offset="95%" stopColor="#c2410c" stopOpacity={0} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#d6cfbe" />
                        <XAxis
                          dataKey="dateLabel"
                          stroke="#6b655b"
                          style={{ fontSize: 11 }}
                          tick={{ fontSize: 11 }}
                        />
                        <YAxis stroke="#6b655b" style={{ fontSize: 11 }} allowDecimals={false} />
                        <Tooltip
                          contentStyle={{
                            background: "#fbf7ee",
                            border: "1px solid #d6cfbe",
                            borderRadius: 2,
                            fontSize: 12,
                            fontFamily: "var(--f-mono)",
                          }}
                        />
                        <Area
                          type="monotone"
                          dataKey="count"
                          stroke="#c2410c"
                          strokeWidth={2}
                          fill="url(#inferenceGrad)"
                          name="Inferences"
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <p className="c-muted" style={{ textAlign: "center", padding: 40 }}>
                    No inference data for this period. Run some inferences to see stats.
                  </p>
                )}
              </div>

              {/* Fraud detection trend */}
              <div className="c-card">
                <h3
                  className="c-h3"
                  style={{ marginBottom: 16, display: "flex", alignItems: "center", gap: 8 }}
                >
                  <Shield size={16} style={{ color: "var(--warn)" }} />
                  Fraud Detection Trend
                </h3>
                {fraudChartData.length > 0 ? (
                  <div style={{ width: "100%", height: 260 }}>
                    <ResponsiveContainer>
                      <LineChart
                        data={fraudChartData}
                        margin={{ top: 5, right: 5, left: -10, bottom: 5 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#d6cfbe" />
                        <XAxis dataKey="dateLabel" stroke="#6b655b" style={{ fontSize: 11 }} />
                        <YAxis stroke="#6b655b" style={{ fontSize: 11 }} domain={[0, 1]} />
                        <Tooltip
                          formatter={(v: number, name: string) => {
                            if (name === "avg_probability")
                              return [(v * 100).toFixed(1) + "%", "Avg Probability"];
                            return [v, name];
                          }}
                          contentStyle={{
                            background: "#fbf7ee",
                            border: "1px solid #d6cfbe",
                            borderRadius: 2,
                            fontSize: 12,
                            fontFamily: "var(--f-mono)",
                          }}
                        />
                        <Legend wrapperStyle={{ fontSize: 11 }} />
                        <Line
                          type="monotone"
                          dataKey="total"
                          stroke="#6b655b"
                          strokeWidth={1.5}
                          dot={false}
                          name="Total checks"
                        />
                        <Line
                          type="monotone"
                          dataKey="fraud_count"
                          stroke="#b91c1c"
                          strokeWidth={2}
                          dot={false}
                          name="Flagged"
                        />
                        <Line
                          type="monotone"
                          dataKey="avg_probability"
                          stroke="#b45309"
                          strokeWidth={1.5}
                          strokeDasharray="4 3"
                          dot={false}
                          name="Avg probability"
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <p className="c-muted" style={{ textAlign: "center", padding: 40 }}>
                    No fraud data for this period. Run fraud checks to see trends.
                  </p>
                )}
              </div>
            </div>

            {/* Row 2: Latency distribution + Model usage pie */}
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(380px, 1fr))",
                gap: 20,
                marginBottom: 24,
              }}
            >
              {/* Latency distribution */}
              <div className="c-card">
                <h3
                  className="c-h3"
                  style={{ marginBottom: 16, display: "flex", alignItems: "center", gap: 8 }}
                >
                  <Zap size={16} style={{ color: "var(--forge)" }} />
                  Latency Distribution
                </h3>
                {latencyBins.length > 0 ? (
                  <div style={{ width: "100%", height: 260 }}>
                    <ResponsiveContainer>
                      <BarChart
                        data={latencyBins}
                        margin={{ top: 5, right: 5, left: -10, bottom: 5 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#d6cfbe" />
                        <XAxis dataKey="range" stroke="#6b655b" style={{ fontSize: 11 }} />
                        <YAxis stroke="#6b655b" style={{ fontSize: 11 }} allowDecimals={false} />
                        <Tooltip
                          formatter={(v: number) => [`${v} inferences`, "Count"]}
                          contentStyle={{
                            background: "#fbf7ee",
                            border: "1px solid #d6cfbe",
                            borderRadius: 2,
                            fontSize: 12,
                            fontFamily: "var(--f-mono)",
                          }}
                        />
                        <Bar dataKey="count" fill="#c2410c" radius={[2, 2, 0, 0]} name="Inferences">
                          {latencyBins.map((_, i) => (
                            <Cell key={i} fill={COLORS[i % COLORS.length]} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <p className="c-muted" style={{ textAlign: "center", padding: 40 }}>
                    No latency data available. Run inferences to see distribution.
                  </p>
                )}
              </div>

              {/* Model usage pie */}
              <div className="c-card">
                <h3
                  className="c-h3"
                  style={{ marginBottom: 16, display: "flex", alignItems: "center", gap: 8 }}
                >
                  <Cpu size={16} style={{ color: "var(--trace)" }} />
                  Model Usage Breakdown
                </h3>
                {modelPieData.length > 0 ? (
                  <div style={{ width: "100%", height: 260 }}>
                    <ResponsiveContainer>
                      <PieChart>
                        <Pie
                          data={modelPieData}
                          dataKey="value"
                          nameKey="name"
                          cx="50%"
                          cy="50%"
                          outerRadius={90}
                          label={({ name, percent }) => `${name} (${(percent * 100).toFixed(0)}%)`}
                          labelLine={{ stroke: "#6b655b", strokeWidth: 0.5 }}
                          style={{ fontSize: 11 }}
                        >
                          {modelPieData.map((_, i) => (
                            <Cell
                              key={i}
                              fill={COLORS[i % COLORS.length]}
                              stroke="#fbf7ee"
                              strokeWidth={1}
                            />
                          ))}
                        </Pie>
                        <Tooltip
                          formatter={(v: number) => [`${v} inferences`, "Usage"]}
                          contentStyle={{
                            background: "#fbf7ee",
                            border: "1px solid #d6cfbe",
                            borderRadius: 2,
                            fontSize: 12,
                            fontFamily: "var(--f-mono)",
                          }}
                        />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <p className="c-muted" style={{ textAlign: "center", padding: 40 }}>
                    No model usage data yet. Upload and run models to see analytics.
                  </p>
                )}
              </div>
            </div>

            {/* Fraud history table */}
            <div className="c-card" style={{ marginBottom: 24 }}>
              <h3 className="c-h3" style={{ marginBottom: 16 }}>
                Recent Fraud Checks
              </h3>
              {fraudHistory.length ? (
                <div style={{ overflowX: "auto" }}>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
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
                          Time
                        </th>
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
                          Type
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
                          Amount
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
                          Probability
                        </th>
                        <th
                          style={{
                            textAlign: "center",
                            padding: "8px 12px",
                            fontSize: 11,
                            textTransform: "uppercase",
                            letterSpacing: ".1em",
                            color: "var(--ink-muted)",
                          }}
                        >
                          Verdict
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {fraudHistory.slice(0, 20).map((f) => (
                        <tr key={f.id} style={{ borderBottom: "1px solid var(--rule)" }}>
                          <td
                            style={{
                              padding: "10px 12px",
                              color: "var(--ink-muted)",
                              fontSize: 12,
                            }}
                          >
                            {new Date(f.created_at).toLocaleString("en-US", {
                              month: "short",
                              day: "numeric",
                              hour: "2-digit",
                              minute: "2-digit",
                            })}
                          </td>
                          <td style={{ padding: "10px 12px" }}>{f.tx_type}</td>
                          <td
                            style={{
                              textAlign: "right",
                              padding: "10px 12px",
                              fontVariantNumeric: "tabular-nums",
                            }}
                          >
                            ₹{fmtNum(f.amount)}
                          </td>
                          <td
                            style={{
                              textAlign: "right",
                              padding: "10px 12px",
                              fontVariantNumeric: "tabular-nums",
                              color:
                                f.probability >= 0.5
                                  ? "var(--risk)"
                                  : f.probability >= 0.35
                                    ? "var(--warn)"
                                    : "var(--ok)",
                            }}
                          >
                            {(f.probability * 100).toFixed(1)}%
                          </td>
                          <td style={{ textAlign: "center", padding: "10px 12px" }}>
                            <span
                              style={{
                                fontSize: 11,
                                padding: "2px 8px",
                                borderRadius: 2,
                                background:
                                  f.probability >= 0.5
                                    ? "#fadcdc"
                                    : f.probability >= 0.35
                                      ? "#f3e7c4"
                                      : "#ddeedf",
                                border:
                                  "1px solid " +
                                  (f.probability >= 0.5
                                    ? "#e4b4b4"
                                    : f.probability >= 0.35
                                      ? "#dcc58a"
                                      : "#b4d8be"),
                                color:
                                  f.probability >= 0.5
                                    ? "var(--risk)"
                                    : f.probability >= 0.35
                                      ? "var(--warn)"
                                      : "var(--ok)",
                                fontFamily: "var(--f-mono)",
                              }}
                            >
                              {f.verdict}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="c-muted" style={{ textAlign: "center", padding: 30 }}>
                  No fraud detection history. Run fraud checks to see data here.
                </p>
              )}
            </div>
          </>
        )}
      </section>

      <style>{`
        .c-spin {
          animation: c-spin 1s linear infinite;
        }
        @keyframes c-spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </CrucibleLayout>
  );
}
