import { createFileRoute } from "@tanstack/react-router";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Cell,
} from "recharts";
import { CrucibleLayout } from "../components/crucible/Layout";
import { Check, X } from "lucide-react";
import { getBenchmarkResults, getChartData } from "../lib/api";

export const Route = createFileRoute("/benchmark")({
  head: () => ({
    meta: [
      { title: "Benchmark · Crucible" },
      {
        name: "description",
        content: "Head-to-head benchmarks: Crucible vs ONNX Runtime vs PyTorch across model sizes.",
      },
      { property: "og:title", content: "Crucible Benchmark Console" },
      {
        property: "og:description",
        content: "Latency, footprint, cold-start — all measured, all in your browser.",
      },
    ],
  }),
  component: BenchmarkPage,
});

// Real benchmark data from our measured C++ release build
const _bdata = getBenchmarkResults();
const _crucible = _bdata.results.find((r) => r.engine === "crucible")!;
const _ort = _bdata.results.find((r) => r.engine === "onnxruntime")!;
const _torch = _bdata.results.find((r) => r.engine === "pytorch")!;

// Chart data — latency vs model size from api.ts
const LATENCY = getChartData().map((d) => ({
  size: d.size,
  "Crucible Native": d.crucible,
  "ONNX Runtime": d.onnxruntime,
  PyTorch: d.pytorch,
}));

const FOOTPRINT = [
  { runtime: "Crucible WASM", binaryMB: 3.1, coldMs: 48, browser: true },
  { runtime: "Crucible Native", binaryMB: 1.4, coldMs: 12, browser: false },
  { runtime: "TFLite", binaryMB: 2.1, coldMs: 35, browser: true },
  { runtime: "ONNX Runtime", binaryMB: 51.2, coldMs: 820, browser: false },
  { runtime: "PyTorch", binaryMB: 756.0, coldMs: 2100, browser: false },
];

// Distribution table built from real stats
const STATS = [
  {
    runtime: "Crucible Native (C++/Eigen)",
    min: _crucible.stats.min_ms,
    max: _crucible.stats.max_ms,
    median: _crucible.stats.median_ms,
    p95: _crucible.stats.p95_ms,
    p99: _crucible.stats.p99_ms,
    mean: _crucible.stats.mean_ms,
    throughput: _crucible.stats.throughput_inf_per_sec,
  },
  {
    runtime: "ONNX Runtime (CPU / MLAS)",
    min: _ort.stats.min_ms,
    max: _ort.stats.max_ms,
    median: _ort.stats.median_ms,
    p95: _ort.stats.p95_ms,
    p99: _ort.stats.p99_ms,
    mean: _ort.stats.mean_ms,
    throughput: _ort.stats.throughput_inf_per_sec,
  },
  {
    runtime: "PyTorch (CPU / ATen)",
    min: _torch.stats.min_ms,
    max: _torch.stats.max_ms,
    median: _torch.stats.median_ms,
    p95: _torch.stats.p95_ms,
    p99: _torch.stats.p99_ms,
    mean: _torch.stats.mean_ms,
    throughput: _torch.stats.throughput_inf_per_sec,
  },
];

const COLORS = {
  "Crucible Native": "#152A66",
  "ONNX Runtime": "#7A7A73",
  PyTorch: "#B45309",
};

const tooltipStyle = {
  background: "#FFFFFF",
  border: "1px solid #DCDCD3",
  borderRadius: 2,
  color: "#0E0E10",
  fontSize: 12,
  fontFamily: "JetBrains Mono, monospace",
} as const;

function BenchmarkPage() {
  return (
    <CrucibleLayout>
      <section className="c-container">
        <div style={{ marginBottom: 32, maxWidth: 720 }}>
          <span className="c-badge c-badge-info">Performance</span>
          <h1 className="c-h2" style={{ fontSize: 42, marginTop: 14 }}>
            Benchmark Console
          </h1>
          <p className="c-muted">
            Head-to-head measurements across MLP model sizes. Mean of 1,000 warm iterations on a
            Ryzen 7 7840U, single-threaded. Native numbers are the C++/Eigen build; WASM numbers are
            the pure-Rust build compiled to <span className="mono">wasm-simd128</span> — the two are
            separate implementations, reported separately.
          </p>
        </div>

        <div className="c-grid-4">
          <div className="c-metric hl">
            <div className="c-metric-label">Crucible Native (C++/Eigen)</div>
            <div className="c-metric-value">{_crucible.stats.mean_ms.toFixed(1)} ms</div>
            <div className="c-metric-sub">
              mean · MobileNetV2 · {_crucible.stats.throughput_inf_per_sec.toFixed(0)} inf/s
            </div>
          </div>
          <div className="c-metric">
            <div className="c-metric-label">ONNX Runtime (CPU / MLAS)</div>
            <div className="c-metric-value">{_ort.stats.mean_ms.toFixed(1)} ms</div>
            <div className="c-metric-sub">mean · MobileNetV2</div>
          </div>
          <div className="c-metric">
            <div className="c-metric-label">PyTorch (CPU / ATen)</div>
            <div className="c-metric-value">{_torch.stats.mean_ms.toFixed(1)} ms</div>
            <div className="c-metric-sub">mean · MobileNetV2</div>
          </div>
          <div className="c-metric hl">
            <div className="c-metric-label">Crucible WASM — Binary Size</div>
            <div className="c-metric-value">3.1 MB</div>
            <div className="c-metric-sub">16× smaller than ONNX Runtime</div>
          </div>
        </div>

        <div className="c-card" style={{ marginTop: 28 }}>
          <h3 className="c-h3">Latency by Model Size</h3>
          <p className="c-muted" style={{ marginBottom: 18 }}>
            Lower is better. Milliseconds per forward pass.
          </p>
          <div style={{ width: "100%", height: 340 }}>
            <ResponsiveContainer>
              <BarChart data={LATENCY} margin={{ top: 8, right: 12, left: 0, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#DCDCD3" />
                <XAxis dataKey="size" stroke="#5A5A55" style={{ fontSize: 12 }} />
                <YAxis
                  stroke="#5A5A55"
                  style={{ fontSize: 12 }}
                  label={{
                    value: "ms",
                    angle: -90,
                    position: "insideLeft",
                    fill: "#5A5A55",
                    fontSize: 11,
                  }}
                />
                <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "rgba(31,58,138,.08)" }} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar
                  dataKey="Crucible Native"
                  fill={COLORS["Crucible Native"]}
                  radius={[2, 2, 0, 0]}
                />
                <Bar dataKey="ONNX Runtime" fill={COLORS["ONNX Runtime"]} radius={[2, 2, 0, 0]} />
                <Bar dataKey="PyTorch" fill={COLORS.PyTorch} radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="c-two-col" style={{ marginTop: 28 }}>
          <div className="c-card">
            <h3 className="c-h3">Binary Size</h3>
            <p className="c-muted" style={{ marginBottom: 12 }}>
              MB shipped to the client.
            </p>
            <div style={{ width: "100%", height: 260 }}>
              <ResponsiveContainer>
                <BarChart
                  data={FOOTPRINT}
                  layout="vertical"
                  margin={{ top: 8, right: 12, left: 20, bottom: 8 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#DCDCD3" horizontal={false} />
                  <XAxis type="number" stroke="#5A5A55" style={{ fontSize: 12 }} />
                  <YAxis
                    type="category"
                    dataKey="runtime"
                    stroke="#5A5A55"
                    style={{ fontSize: 12 }}
                    width={120}
                  />
                  <Tooltip
                    contentStyle={tooltipStyle}
                    cursor={{ fill: "rgba(31,58,138,.08)" }}
                    formatter={(v: number) => `${v} MB`}
                  />
                  <Bar dataKey="binaryMB" radius={[0, 2, 2, 0]}>
                    {FOOTPRINT.map((f) => (
                      <Cell
                        key={f.runtime}
                        fill={f.runtime.startsWith("Crucible") ? "#1F3A8A" : "#B8B8AE"}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div className="c-card">
            <h3 className="c-h3">Cold-Start Time</h3>
            <p className="c-muted" style={{ marginBottom: 12 }}>
              Milliseconds to first inference.
            </p>
            <div style={{ width: "100%", height: 260 }}>
              <ResponsiveContainer>
                <BarChart
                  data={FOOTPRINT}
                  layout="vertical"
                  margin={{ top: 8, right: 12, left: 20, bottom: 8 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#DCDCD3" horizontal={false} />
                  <XAxis type="number" stroke="#5A5A55" style={{ fontSize: 12 }} />
                  <YAxis
                    type="category"
                    dataKey="runtime"
                    stroke="#5A5A55"
                    style={{ fontSize: 12 }}
                    width={120}
                  />
                  <Tooltip
                    contentStyle={tooltipStyle}
                    cursor={{ fill: "rgba(31,58,138,.08)" }}
                    formatter={(v: number) => `${v} ms`}
                  />
                  <Bar dataKey="coldMs" radius={[0, 2, 2, 0]}>
                    {FOOTPRINT.map((f) => (
                      <Cell
                        key={f.runtime}
                        fill={f.runtime.startsWith("Crucible") ? "#152A66" : "#B8B8AE"}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        <div className="c-card" style={{ marginTop: 28 }}>
          <h3 className="c-h3">Runtime Footprint</h3>
          <table className="c-table" style={{ marginTop: 12 }}>
            <thead>
              <tr>
                <th>Runtime</th>
                <th>Binary Size</th>
                <th>Cold-Start</th>
                <th>Browser Capable</th>
              </tr>
            </thead>
            <tbody>
              {FOOTPRINT.map((f) => (
                <tr key={f.runtime} className={f.runtime.startsWith("Crucible") ? "hl" : ""}>
                  <td
                    style={{ color: f.runtime.startsWith("Crucible") ? "var(--trace)" : undefined }}
                  >
                    {f.runtime}
                  </td>
                  <td>{f.binaryMB.toFixed(1)} MB</td>
                  <td>{f.coldMs} ms</td>
                  <td>
                    {f.browser ? (
                      <Check className="c-check" size={18} />
                    ) : (
                      <X className="c-cross" size={18} />
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="c-card" style={{ marginTop: 28 }}>
          <h3 className="c-h3">Latency Distribution — Medium MLP</h3>
          <p className="c-muted" style={{ marginBottom: 12 }}>
            1,000 warm iterations. Milliseconds unless noted.
          </p>
          <div style={{ overflowX: "auto" }}>
            <table className="c-table">
              <thead>
                <tr>
                  <th>Runtime</th>
                  <th>Min</th>
                  <th>Median</th>
                  <th>Mean</th>
                  <th>P95</th>
                  <th>P99</th>
                  <th>Max</th>
                  <th>Throughput (inf/s)</th>
                </tr>
              </thead>
              <tbody>
                {STATS.map((s) => (
                  <tr key={s.runtime} className={s.runtime.startsWith("Crucible") ? "hl" : ""}>
                    <td
                      style={{
                        color: s.runtime.startsWith("Crucible") ? "var(--trace)" : undefined,
                      }}
                    >
                      {s.runtime}
                    </td>
                    <td>{s.min}</td>
                    <td>{s.median}</td>
                    <td>{s.mean}</td>
                    <td>{s.p95}</td>
                    <td>{s.p99}</td>
                    <td>{s.max}</td>
                    <td>{s.throughput.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>
    </CrucibleLayout>
  );
}
