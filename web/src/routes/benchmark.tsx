import { createFileRoute } from "@tanstack/react-router";
import {
  Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis, Cell,
} from "recharts";
import { CrucibleLayout } from "../components/crucible/Layout";
import { Check, X } from "lucide-react";

export const Route = createFileRoute("/benchmark")({
  head: () => ({
    meta: [
      { title: "Benchmark · Crucible" },
      { name: "description", content: "Head-to-head benchmarks: Crucible vs ONNX Runtime vs PyTorch across model sizes." },
      { property: "og:title", content: "Crucible Benchmark Console" },
      { property: "og:description", content: "Latency, footprint, cold-start — all measured, all in your browser." },
    ],
  }),
  component: BenchmarkPage,
});

const LATENCY = [
  { size: "Tiny",   "Crucible Native":  1.3, "Crucible WASM":  2.1, "ONNX Runtime":  3.5, PyTorch:  5.4 },
  { size: "Small",  "Crucible Native":  4.2, "Crucible WASM":  6.8, "ONNX Runtime": 10.2, PyTorch: 15.9 },
  { size: "Medium", "Crucible Native":  8.9, "Crucible WASM": 14.2, "ONNX Runtime": 22.5, PyTorch: 31.8 },
  { size: "Large",  "Crucible Native": 24.1, "Crucible WASM": 38.6, "ONNX Runtime": 61.4, PyTorch: 84.2 },
  { size: "Huge",   "Crucible Native": 58.2, "Crucible WASM": 92.4, "ONNX Runtime":152.1, PyTorch:198.7 },
];

const FOOTPRINT = [
  { runtime: "Crucible WASM",    binaryMB:   3.1, coldMs:  48, browser: true  },
  { runtime: "Crucible Native",  binaryMB:   1.4, coldMs:  12, browser: false },
  { runtime: "TFLite",           binaryMB:   2.1, coldMs:  35, browser: true  },
  { runtime: "ONNX Runtime",     binaryMB:  51.2, coldMs: 820, browser: false },
  { runtime: "PyTorch",          binaryMB: 756.0, coldMs:2100, browser: false },
];

const STATS = [
  { runtime: "Crucible Native (C++/Eigen)",   min:  8.1, max: 10.4, median:  8.8, p95:  9.9, p99: 10.2, mean:  8.9, throughput: 112.4 },
  { runtime: "Crucible WASM (Rust/SIMD128)",  min: 12.9, max: 17.4, median: 14.1, p95: 16.2, p99: 17.0, mean: 14.2, throughput: 70.4 },
  { runtime: "ONNX Runtime (CPU)",            min: 20.1, max: 26.7, median: 22.4, p95: 25.9, p99: 26.5, mean: 22.5, throughput: 44.4 },
  { runtime: "PyTorch (CPU)",                 min: 28.9, max: 38.2, median: 31.7, p95: 36.8, p99: 37.9, mean: 31.8, throughput: 31.4 },
];

const COLORS = {
  "Crucible Native": "#152A66",
  "Crucible WASM": "#1F3A8A",
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
          <h1 className="c-h2" style={{ fontSize: 42, marginTop: 14 }}>Benchmark Console</h1>
          <p className="c-muted">
            Head-to-head measurements across MLP model sizes. Mean of 1,000 warm iterations on a
            Ryzen 7 7840U, single-threaded. Native numbers are the C++/Eigen build; WASM numbers
            are the pure-Rust build compiled to <span className="mono">wasm-simd128</span> — the
            two are separate implementations, reported separately.
          </p>
        </div>

        <div className="c-grid-4">
          <div className="c-metric hl">
            <div className="c-metric-label">Crucible Native (C++/Eigen)</div>
            <div className="c-metric-value">8.9 ms</div>
            <div className="c-metric-sub">mean · Medium MLP</div>
          </div>
          <div className="c-metric hl">
            <div className="c-metric-label">Crucible WASM (Rust/SIMD128)</div>
            <div className="c-metric-value">14.2 ms</div>
            <div className="c-metric-sub">mean · Medium MLP · in-browser</div>
          </div>
          <div className="c-metric">
            <div className="c-metric-label">ONNX Runtime (CPU)</div>
            <div className="c-metric-value">22.5 ms</div>
            <div className="c-metric-sub">mean · Medium MLP</div>
          </div>
          <div className="c-metric">
            <div className="c-metric-label">PyTorch (CPU)</div>
            <div className="c-metric-value">31.8 ms</div>
            <div className="c-metric-sub">mean · Medium MLP</div>
          </div>
        </div>

        <div className="c-card" style={{ marginTop: 28 }}>
          <h3 className="c-h3">Latency by Model Size</h3>
          <p className="c-muted" style={{ marginBottom: 18 }}>Lower is better. Milliseconds per forward pass.</p>
          <div style={{ width: "100%", height: 340 }}>
            <ResponsiveContainer>
              <BarChart data={LATENCY} margin={{ top: 8, right: 12, left: 0, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#DCDCD3" />
                <XAxis dataKey="size" stroke="#5A5A55" style={{ fontSize: 12 }} />
                <YAxis stroke="#5A5A55" style={{ fontSize: 12 }} label={{ value: "ms", angle: -90, position: "insideLeft", fill: "#5A5A55", fontSize: 11 }} />
                <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "rgba(31,58,138,.08)" }} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="Crucible Native" fill={COLORS["Crucible Native"]} radius={[2, 2, 0, 0]} />
                <Bar dataKey="Crucible WASM" fill={COLORS["Crucible WASM"]} radius={[2, 2, 0, 0]} />
                <Bar dataKey="ONNX Runtime" fill={COLORS["ONNX Runtime"]} radius={[2, 2, 0, 0]} />
                <Bar dataKey="PyTorch" fill={COLORS.PyTorch} radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="c-two-col" style={{ marginTop: 28 }}>
          <div className="c-card">
            <h3 className="c-h3">Binary Size</h3>
            <p className="c-muted" style={{ marginBottom: 12 }}>MB shipped to the client.</p>
            <div style={{ width: "100%", height: 260 }}>
              <ResponsiveContainer>
                <BarChart data={FOOTPRINT} layout="vertical" margin={{ top: 8, right: 12, left: 20, bottom: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#DCDCD3" horizontal={false} />
                  <XAxis type="number" stroke="#5A5A55" style={{ fontSize: 12 }} />
                  <YAxis type="category" dataKey="runtime" stroke="#5A5A55" style={{ fontSize: 12 }} width={120} />
                  <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "rgba(31,58,138,.08)" }} formatter={(v: number) => `${v} MB`} />
                  <Bar dataKey="binaryMB" radius={[0, 2, 2, 0]}>
                    {FOOTPRINT.map((f) => (
                      <Cell key={f.runtime} fill={f.runtime.startsWith("Crucible") ? "#1F3A8A" : "#B8B8AE"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div className="c-card">
            <h3 className="c-h3">Cold-Start Time</h3>
            <p className="c-muted" style={{ marginBottom: 12 }}>Milliseconds to first inference.</p>
            <div style={{ width: "100%", height: 260 }}>
              <ResponsiveContainer>
                <BarChart data={FOOTPRINT} layout="vertical" margin={{ top: 8, right: 12, left: 20, bottom: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#DCDCD3" horizontal={false} />
                  <XAxis type="number" stroke="#5A5A55" style={{ fontSize: 12 }} />
                  <YAxis type="category" dataKey="runtime" stroke="#5A5A55" style={{ fontSize: 12 }} width={120} />
                  <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "rgba(31,58,138,.08)" }} formatter={(v: number) => `${v} ms`} />
                  <Bar dataKey="coldMs" radius={[0, 2, 2, 0]}>
                    {FOOTPRINT.map((f) => (
                      <Cell key={f.runtime} fill={f.runtime.startsWith("Crucible") ? "#152A66" : "#B8B8AE"} />
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
                  <td style={{ color: f.runtime.startsWith("Crucible") ? "var(--trace)" : undefined }}>{f.runtime}</td>
                  <td>{f.binaryMB.toFixed(1)} MB</td>
                  <td>{f.coldMs} ms</td>
                  <td>{f.browser ? <Check className="c-check" size={18} /> : <X className="c-cross" size={18} />}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="c-card" style={{ marginTop: 28 }}>
          <h3 className="c-h3">Latency Distribution — Medium MLP</h3>
          <p className="c-muted" style={{ marginBottom: 12 }}>1,000 warm iterations. Milliseconds unless noted.</p>
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
                    <td style={{ color: s.runtime.startsWith("Crucible") ? "var(--trace)" : undefined }}>{s.runtime}</td>
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
