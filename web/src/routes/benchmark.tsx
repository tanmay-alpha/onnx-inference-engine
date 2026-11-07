import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Check, X, ExternalLink } from "lucide-react";
import { CrucibleLayout } from "../components/crucible/Layout";
import { getBenchmarkFile, type BenchmarkFile, type BenchmarkResult } from "../lib/api";

export const Route = createFileRoute("/benchmark")({
  head: () => ({
    meta: [
      { title: "Benchmark · Crucible" },
      {
        name: "description",
        content:
          "Measured latency of the Crucible native engine against ONNX Runtime and PyTorch, with environment and methodology.",
      },
      { property: "og:title", content: "Crucible Benchmarks" },
      {
        property: "og:description",
        content: "Measured results from benchmarks/run_all.py — nothing estimated.",
      },
    ],
  }),
  component: BenchmarkPage,
});

const METHODOLOGY_URL =
  "https://github.com/tanmay-alpha/onnx-inference-engine/blob/main/docs/BENCHMARKS.md";

const RUNTIME_LABEL: Record<string, string> = {
  "crucible-native": "Crucible native (C++)",
  "crucible-native-avx2": "Crucible native (C++, AVX2)",
  "onnxruntime-cpu": "ONNX Runtime CPU",
  "onnxruntime-cpu-1t": "ONNX Runtime CPU (1 thread)",
  "pytorch-cpu": "PyTorch CPU",
  "crucible-wasm": "Crucible WASM (Rust)",
};

const RUNTIME_COLOR: Record<string, string> = {
  "crucible-native": "#152A66",
  "crucible-native-avx2": "#1F3A8A",
  "onnxruntime-cpu": "#7A7A73",
  "onnxruntime-cpu-1t": "#A8A89E",
  "pytorch-cpu": "#B45309",
  "crucible-wasm": "#C2410C",
};

const tooltipStyle = {
  background: "#FFFFFF",
  border: "1px solid #DCDCD3",
  borderRadius: 2,
  color: "#0E0E10",
  fontSize: 12,
  fontFamily: "JetBrains Mono, monospace",
} as const;

const label = (runtime: string) => RUNTIME_LABEL[runtime] ?? runtime;
const ms = (v: number | null) => (v === null ? "—" : v < 1 ? v.toFixed(3) : v.toFixed(2));
const fmtBytes = (b: number) =>
  b < 1024
    ? `${b} B`
    : b < 1024 * 1024
      ? `${(b / 1024).toFixed(1)} KB`
      : `${(b / 1048576).toFixed(2)} MB`;

/** One row group per (model, input_shape). */
function groupResults(results: BenchmarkResult[]) {
  const groups = new Map<string, { model: string; shape: number[]; rows: BenchmarkResult[] }>();
  for (const r of results) {
    const key = `${r.model} [${r.input_shape.join("×")}]`;
    if (!groups.has(key)) groups.set(key, { model: r.model, shape: r.input_shape, rows: [] });
    groups.get(key)!.rows.push(r);
  }
  return Array.from(groups, ([key, g]) => ({ key, ...g }));
}

/** Sentence comparing Crucible native with ONNX Runtime, built only from measured medians. */
function comparison(rows: BenchmarkResult[]): string | null {
  const ok = (rt: string) =>
    rows.find((r) => r.runtime === rt && r.status === "ok" && r.median_ms !== null);
  const crucible = ok("crucible-native-avx2") ?? ok("crucible-native");
  const ort = ok("onnxruntime-cpu-1t") ?? ok("onnxruntime-cpu");
  if (!crucible || !ort) return null;
  const ratio = crucible.median_ms! / ort.median_ms!;
  const rel =
    ratio >= 1 ? `${ratio.toFixed(1)}× slower than` : `${(1 / ratio).toFixed(1)}× faster than`;
  const threads =
    crucible.threads === ort.threads
      ? `both ${ort.threads} thread${ort.threads === 1 ? "" : "s"}`
      : `${crucible.threads} vs ${ort.threads} threads`;
  return `${label(crucible.runtime)} median ${ms(crucible.median_ms)} ms is ${rel} ${label(ort.runtime)} at ${ms(ort.median_ms)} ms (${threads}).`;
}

function BenchmarkPage() {
  const [state, setState] = useState<
    | { kind: "loading" }
    | { kind: "missing" }
    | { kind: "error"; message: string }
    | { kind: "ok"; data: BenchmarkFile }
  >({ kind: "loading" });

  useEffect(() => {
    getBenchmarkFile().then(
      (data) => setState(data ? { kind: "ok", data } : { kind: "missing" }),
      (e: unknown) =>
        setState({ kind: "error", message: e instanceof Error ? e.message : String(e) }),
    );
  }, []);

  const data = state.kind === "ok" ? state.data : null;
  const groups = useMemo(() => (data ? groupResults(data.results) : []), [data]);
  const runtimes = useMemo(
    () => (data ? Array.from(new Set(data.results.map((r) => r.runtime))) : []),
    [data],
  );
  const chartData = useMemo(
    () =>
      groups.map((g) => {
        const point: Record<string, string | number | null> = { name: g.key };
        for (const r of g.rows) point[r.runtime] = r.status === "ok" ? r.median_ms : null;
        return point;
      }),
    [groups],
  );

  return (
    <CrucibleLayout>
      <section className="c-container">
        <div style={{ marginBottom: 32, maxWidth: 760 }}>
          <span className="c-badge c-badge-info">Performance</span>
          <h1 className="c-h2" style={{ fontSize: 42, marginTop: 14 }}>
            Benchmark Console
          </h1>
          <p className="c-muted">
            Every number on this page comes from{" "}
            <span className="mono">web/public/benchmarks/results.json</span>, written by{" "}
            <span className="mono">python benchmarks/run_all.py</span>. Nothing is estimated or
            hand-entered. Methodology and caveats:{" "}
            <a
              href={METHODOLOGY_URL}
              target="_blank"
              rel="noreferrer noopener"
              style={{ color: "var(--trace)" }}
            >
              docs/BENCHMARKS.md <ExternalLink size={11} style={{ verticalAlign: -1 }} />
            </a>
            .
          </p>
        </div>

        {state.kind === "loading" && <div className="c-loading-bar" />}

        {state.kind === "error" && (
          <div role="alert" className="c-card" style={{ color: "var(--risk)" }}>
            {state.message}
          </div>
        )}

        {state.kind === "missing" && (
          <div className="c-card c-card-dashed" style={{ padding: 40, textAlign: "center" }}>
            <h3 className="c-h3">No benchmark results published yet</h3>
            <p className="c-muted" style={{ maxWidth: 560, margin: "10px auto 0" }}>
              This deployment has no <span className="mono">/benchmarks/results.json</span>. Build
              the native engine, then run the suite from the repository root:
            </p>
            <pre className="c-code" style={{ display: "inline-block", marginTop: 14 }}>
              python benchmarks/run_all.py
            </pre>
            <p className="c-muted" style={{ marginTop: 10, fontSize: 13 }}>
              See{" "}
              <a
                href={METHODOLOGY_URL}
                target="_blank"
                rel="noreferrer noopener"
                style={{ color: "var(--trace)" }}
              >
                docs/BENCHMARKS.md
              </a>{" "}
              for the methodology.
            </p>
          </div>
        )}

        {data && (
          <>
            <div className="c-two-col">
              <div className="c-card">
                <h3 className="c-h3">Environment</h3>
                <div className="c-spec" style={{ marginTop: 12 }}>
                  {(
                    [
                      ["cpu", `${data.environment.cpu} (${data.environment.cores} cores)`],
                      ["os", data.environment.os],
                      ["python", data.environment.python],
                      ["compiler", data.environment.compiler],
                      ["crucible build", data.environment.crucible_build],
                      ["onnxruntime", data.environment.onnxruntime],
                      ["torch", data.environment.torch ?? "not installed"],
                      ["git commit", data.git_commit],
                      ["generated", new Date(data.generated_at).toLocaleString()],
                    ] as [string, string][]
                  ).map(([k, v]) => (
                    <div className="c-spec-row" key={k}>
                      <div className="k">{k}</div>
                      <div className="v">{v}</div>
                    </div>
                  ))}
                </div>
              </div>
              <div className="c-card">
                <h3 className="c-h3">Methodology</h3>
                <p className="c-muted" style={{ marginTop: 12 }}>
                  {data.methodology.warmup} warm-up runs, then {data.methodology.iterations} timed
                  iterations per configuration. Timer:{" "}
                  <span className="mono">{data.methodology.timer}</span>.
                </p>
                {data.methodology.notes.length > 0 && (
                  <ul className="c-muted" style={{ margin: "10px 0 0 18px", fontSize: 13 }}>
                    {data.methodology.notes.map((n) => (
                      <li key={n} style={{ marginTop: 4 }}>
                        {n}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>

            {chartData.length > 0 && (
              <div className="c-card" style={{ marginTop: 28 }}>
                <h3 className="c-h3">Median latency by model</h3>
                <p className="c-muted" style={{ marginBottom: 18 }}>
                  Milliseconds per forward pass, lower is better. Missing bars are unsupported
                  configurations.
                </p>
                <div style={{ width: "100%", height: 340 }}>
                  <ResponsiveContainer>
                    <BarChart data={chartData} margin={{ top: 8, right: 12, left: 0, bottom: 8 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#DCDCD3" />
                      <XAxis dataKey="name" stroke="#5A5A55" style={{ fontSize: 11 }} />
                      <YAxis stroke="#5A5A55" style={{ fontSize: 12 }} />
                      <Tooltip
                        contentStyle={tooltipStyle}
                        cursor={{ fill: "rgba(31,58,138,.08)" }}
                        formatter={(v: number, name: string) => [`${ms(v)} ms`, label(name)]}
                      />
                      <Legend wrapperStyle={{ fontSize: 12 }} formatter={(v: string) => label(v)} />
                      {runtimes.map((rt) => (
                        <Bar
                          key={rt}
                          dataKey={rt}
                          fill={RUNTIME_COLOR[rt] ?? "#888"}
                          radius={[2, 2, 0, 0]}
                        />
                      ))}
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {groups.map((g) => {
              const summary = comparison(g.rows);
              return (
                <div className="c-card" style={{ marginTop: 28 }} key={g.key}>
                  <h3 className="c-h3">
                    <span className="mono">{g.model}</span>{" "}
                    <span className="c-muted" style={{ fontSize: 14 }}>
                      input [{g.shape.join(", ")}]
                    </span>
                  </h3>
                  {summary && (
                    <p className="c-muted" style={{ marginTop: 6 }}>
                      {summary}
                    </p>
                  )}
                  <div style={{ overflowX: "auto" }}>
                    <table className="c-table" style={{ marginTop: 12 }}>
                      <thead>
                        <tr>
                          <th>Runtime</th>
                          <th>Threads</th>
                          <th>Mean</th>
                          <th>Median</th>
                          <th>p95</th>
                          <th>Min</th>
                          <th>Std</th>
                          <th>Throughput</th>
                          <th>Load</th>
                          <th>Max |Δ| vs ORT</th>
                          <th>Top-1 match</th>
                        </tr>
                      </thead>
                      <tbody>
                        {g.rows.map((r) => (
                          <tr
                            key={`${r.runtime}-${r.threads}`}
                            className={r.runtime.startsWith("crucible") ? "hl" : ""}
                          >
                            <td>
                              {label(r.runtime)}
                              {r.note && (
                                <div className="c-muted" style={{ fontSize: 11 }}>
                                  {r.note}
                                </div>
                              )}
                            </td>
                            <td>{r.threads}</td>
                            {r.status === "unsupported" ? (
                              <td colSpan={9} className="c-muted">
                                unsupported
                              </td>
                            ) : (
                              <>
                                <td>{ms(r.mean_ms)} ms</td>
                                <td>{ms(r.median_ms)} ms</td>
                                <td>{ms(r.p95_ms)} ms</td>
                                <td>{ms(r.min_ms)} ms</td>
                                <td>{ms(r.std_ms)}</td>
                                <td>
                                  {r.throughput_ips === null
                                    ? "—"
                                    : `${r.throughput_ips.toFixed(1)} /s`}
                                </td>
                                <td>{r.load_ms === null ? "—" : `${ms(r.load_ms)} ms`}</td>
                                <td>
                                  {r.max_abs_diff_vs_ort === null
                                    ? "—"
                                    : r.max_abs_diff_vs_ort.toExponential(2)}
                                </td>
                                <td>
                                  {r.top1_match === null ? (
                                    "—"
                                  ) : r.top1_match ? (
                                    <Check className="c-check" size={16} />
                                  ) : (
                                    <X className="c-cross" size={16} />
                                  )}
                                </td>
                              </>
                            )}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              );
            })}

            {data.artifacts.length > 0 && (
              <div className="c-card" style={{ marginTop: 28 }}>
                <h3 className="c-h3">Artifact sizes</h3>
                <table className="c-table" style={{ marginTop: 12 }}>
                  <thead>
                    <tr>
                      <th>Artifact</th>
                      <th>Size</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.artifacts.map((a) => (
                      <tr key={a.name}>
                        <td className="mono">{a.name}</td>
                        <td>{fmtBytes(a.bytes)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </>
        )}
      </section>
    </CrucibleLayout>
  );
}
