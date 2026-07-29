import { createFileRoute, Link } from "@tanstack/react-router";
import { useMemo, useState, useCallback, useEffect, useRef } from "react";
import {
  Play,
  Upload,
  ChevronRight,
  ChevronDown,
  History,
  Settings,
  Server,
  Cpu,
} from "lucide-react";
import { CrucibleLayout } from "../components/crucible/Layout";
import { initWasm, runWasmInference } from "../lib/crucible-wasm";
import { fetchModels, runInference, type RegisteredModel } from "../lib/api";

export const Route = createFileRoute("/playground")({
  head: () => ({
    meta: [
      { title: "Playground · Crucible" },
      {
        name: "description",
        content: "Drop an ONNX model, feed a tensor, inspect the graph — all in the browser.",
      },
      { property: "og:title", content: "Crucible WASM Playground" },
      {
        property: "og:description",
        content: "Interactive ONNX runtime running entirely client-side.",
      },
    ],
  }),
  component: PlaygroundPage,
});

interface GraphNode {
  op: string;
  inputs: string;
  output: string;
  attrs?: Record<string, string>;
}

const DEFAULT_MODEL = {
  name: "fraud_mlp_v3.onnx",
  irVersion: 8,
  opset: 17,
  producer: "pytorch 2.3.0",
  bytes: 220,
  nodes: [
    { op: "MatMul", inputs: "input, W0[7,16]", output: "h0_pre", attrs: { transB: "0" } },
    { op: "Add", inputs: "h0_pre, b0[16]", output: "h0_bias" },
    { op: "Relu", inputs: "h0_bias", output: "h0" },
    { op: "MatMul", inputs: "h0, W1[16,8]", output: "h1_pre" },
    { op: "Add", inputs: "h1_pre, b1[8]", output: "h1_bias" },
    { op: "Relu", inputs: "h1_bias", output: "h1" },
    { op: "MatMul", inputs: "h1, W2[8,1]", output: "logit" },
    { op: "Sigmoid", inputs: "logit", output: "output" },
  ] satisfies GraphNode[],
};

type InferenceMode = "wasm" | "server" | "both";
type SampleMode = "zeros" | "random" | "custom";

interface HistoryEntry {
  id: string;
  timestamp: number;
  model: string;
  mode: InferenceMode;
  values: number[];
  output: number[];
  latency: number;
  shape: number[];
}

function PlaygroundPage() {
  const [modelName, setModelName] = useState(DEFAULT_MODEL.name);
  const [modelBytes, setModelBytes] = useState<Uint8Array | null>(null);
  const [shape, setShape] = useState<number[]>([1, 7]);
  const [values, setValues] = useState("0.31, 0.55, 0.02, 1.0, 0.88, 0.12, 0.44");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<{
    latencyMs: number;
    output: number[];
    shape: number[];
    engine: string;
  } | null>(null);
  const [expanded, setExpanded] = useState<number | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [mode, setMode] = useState<InferenceMode>("wasm");
  const [sampleMode, setSampleMode] = useState<SampleMode>("custom");
  const [serverModels, setServerModels] = useState<RegisteredModel[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string>("");
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  const dropRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const nodes = DEFAULT_MODEL.nodes;

  // Load server models
  useEffect(() => {
    fetchModels().then((data) => {
      setServerModels(data);
      if (data.length > 0 && !selectedModelId) {
        setSelectedModelId(data[0].id);
      }
    });
  }, []);

  const parsedValues = useMemo(
    () =>
      values
        .split(/[\s,]+/)
        .filter(Boolean)
        .map(Number),
    [values],
  );
  const expectedSize = shape.reduce((a, b) => a * b, 1);
  const valid = parsedValues.length === expectedSize && parsedValues.every((v) => !Number.isNaN(v));

  const applySample = useCallback(
    (mode: SampleMode) => {
      setSampleMode(mode);
      const size = shape.reduce((a, b) => a * b, 1);
      if (mode === "zeros") {
        setValues(Array.from({ length: size }, () => "0").join(", "));
      } else if (mode === "random") {
        const vals = Array.from({ length: size }, () => Math.random().toFixed(4));
        setValues(vals.join(", "));
      }
    },
    [shape],
  );

  const handleModelFile = useCallback((f: File) => {
    setModelName(f.name);
    setError(null);
    const reader = new FileReader();
    reader.onload = (ev) => {
      const buf = ev.target?.result;
      if (buf instanceof ArrayBuffer) {
        setModelBytes(new Uint8Array(buf));
      }
    };
    reader.onerror = () => setError(`Failed to read file: ${f.name}`);
    reader.readAsArrayBuffer(f);
  }, []);

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    const f = e.dataTransfer.files[0];
    if (f) handleModelFile(f);
  };

  const runWasm = useCallback(async (): Promise<{
    output: number[];
    shape: number[];
    latency: number;
  }> => {
    if (!modelBytes) {
      throw new Error("No model loaded. Drop an .onnx file or click the drop zone to choose one.");
    }
    const inputData = new Float32Array(parsedValues);
    const t0 = performance.now();
    const output = await runWasmInference(modelBytes, inputData, shape);
    const latency = performance.now() - t0;
    return {
      output: Array.from(output),
      shape: shape.slice(1).concat([output.length]),
      latency: Number(latency.toFixed(3)),
    };
  }, [modelBytes, parsedValues, shape]);

  const runServer = useCallback(async (): Promise<{
    output: number[];
    shape: number[];
    latency: number;
  }> => {
    if (!selectedModelId) {
      throw new Error("No server model selected. Upload a model or select one from the list.");
    }
    const t0 = performance.now();
    const res = await runInference(selectedModelId, parsedValues, shape);
    const latency = performance.now() - t0;
    return {
      output: res.output,
      shape: res.shape,
      latency: Number(latency.toFixed(3)),
    };
  }, [selectedModelId, parsedValues, shape]);

  const runBoth = useCallback(async (): Promise<{
    output: number[];
    shape: number[];
    latency: number;
    engine: string;
  }> => {
    // Try server first, fall back to WASM
    try {
      const serverResult = await runServer();
      return { ...serverResult, engine: "server (C++ backend)" };
    } catch {
      const wasmResult = await runWasm();
      return { ...wasmResult, engine: "wasm-simd128 (fallback)" };
    }
  }, [runServer, runWasm]);

  const run = async () => {
    if (!valid) return;
    setRunning(true);
    setResult(null);
    setError(null);

    // Init WASM for wasm/both modes
    if (mode === "wasm" || mode === "both") {
      try {
        await initWasm();
      } catch (e) {
        console.warn("WASM init failed:", e);
        if (mode === "wasm") {
          setError("WebAssembly runtime unavailable. Try server mode.");
          setRunning(false);
          return;
        }
      }
    }

    try {
      let inferenceResult: { output: number[]; shape: number[]; latency: number; engine: string };
      if (mode === "wasm") {
        const r = await runWasm();
        inferenceResult = { ...r, engine: "wasm-simd128" };
      } else if (mode === "server") {
        const r = await runServer();
        inferenceResult = { ...r, engine: "server (C++ backend)" };
      } else {
        const r = await runBoth();
        inferenceResult = r;
      }

      setResult({
        latencyMs: inferenceResult.latency,
        output: inferenceResult.output,
        shape: inferenceResult.shape,
        engine: inferenceResult.engine,
      });

      // Add to history
      const entry: HistoryEntry = {
        id: crypto.randomUUID(),
        timestamp: Date.now(),
        model: modelName,
        mode,
        values: [...parsedValues],
        output: inferenceResult.output,
        latency: inferenceResult.latency,
        shape: inferenceResult.shape,
      };
      setHistory((prev) => [entry, ...prev].slice(0, 5));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  };

  const updateShape = (i: number, v: number) => {
    setShape((s) =>
      s.map((x, idx) => {
        if (idx !== i) return x;
        const parsed = Number.parseInt(String(v), 10);
        if (Number.isNaN(parsed) || parsed < 1) return 1;
        return parsed;
      }),
    );
  };

  const restoreFromHistory = (entry: HistoryEntry) => {
    setValues(entry.values.map((v) => String(v)).join(", "));
    setShape(entry.shape);
    setModelName(entry.model);
    setResult({
      latencyMs: entry.latency,
      output: entry.output,
      shape: entry.shape,
      engine: entry.mode === "wasm" ? "wasm-simd128 (from history)" : "server (from history)",
    });
    setShowHistory(false);
  };

  return (
    <CrucibleLayout>
      <section className="c-container">
        <div
          style={{
            marginBottom: 32,
            maxWidth: 720,
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-start",
            flexWrap: "wrap",
            gap: 12,
          }}
        >
          <div>
            <span className="c-badge c-badge-info">Developer Console</span>
            <h1 className="c-h2" style={{ fontSize: 44, marginTop: 14 }}>
              WASM Inference Playground
            </h1>
            <p className="c-muted">
              Drop an <span className="mono">.onnx</span> model, define input shape and tensor
              values, then execute a forward pass inside the browser sandbox or via the server.
            </p>
          </div>
          <button
            onClick={() => setShowHistory(!showHistory)}
            className="c-btn c-btn-ghost"
            style={{ display: "inline-flex", alignItems: "center", gap: 6 }}
          >
            <History size={15} />
            History ({history.length})
          </button>
        </div>

        {/* Input history panel */}
        {showHistory && history.length > 0 && (
          <div className="c-card" style={{ marginBottom: 24 }}>
            <h3 className="c-h3" style={{ marginBottom: 12 }}>
              Recent Inferences ({history.length})
            </h3>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {history.map((entry) => (
                <div
                  key={entry.id}
                  className="c-history-item"
                  onClick={() => restoreFromHistory(entry)}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 12,
                    padding: "10px 12px",
                    border: "1px solid var(--rule)",
                    borderRadius: 2,
                    cursor: "pointer",
                    transition: "background 0.12s",
                  }}
                >
                  <span
                    style={{
                      fontSize: 10,
                      padding: "2px 6px",
                      borderRadius: 2,
                      background: entry.mode === "wasm" ? "#ddeedf" : "#fce8d5",
                      color: entry.mode === "wasm" ? "#166534" : "#9a3412",
                      fontFamily: "var(--f-mono)",
                      flexShrink: 0,
                    }}
                  >
                    {entry.mode.toUpperCase()}
                  </span>
                  <span style={{ fontFamily: "var(--f-mono)", fontSize: 12, color: "var(--ink)" }}>
                    {entry.model}
                  </span>
                  <span className="c-muted" style={{ fontSize: 11, flex: 1 }}>
                    [{entry.shape.join(", ")}]
                  </span>
                  <span
                    className="mono"
                    style={{ fontSize: 12, color: "var(--forge)", flexShrink: 0 }}
                  >
                    {entry.latency}ms
                  </span>
                  <span
                    className="mono"
                    style={{
                      fontSize: 12,
                      color: entry.output[0] >= 0.5 ? "var(--risk)" : "var(--ok)",
                      flexShrink: 0,
                    }}
                  >
                    {(entry.output[0] * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="c-two-col">
          {/* LEFT — inference console */}
          <div className="c-card">
            <h3 className="c-h3">Inference Console</h3>

            {/* Mode selector */}
            <div style={{ marginTop: 12 }}>
              <label className="c-label">Inference Mode</label>
              <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                {(
                  [
                    { value: "wasm", label: "WASM (Browser)", icon: <Cpu size={13} /> },
                    { value: "server", label: "Server (C++)", icon: <Server size={13} /> },
                    { value: "both", label: "Auto-fallback", icon: <Settings size={13} /> },
                  ] as { value: InferenceMode; label: string; icon: React.ReactNode }[]
                ).map((m) => (
                  <button
                    key={m.value}
                    className={`c-preset ${mode === m.value ? "active" : ""}`}
                    style={{
                      width: "auto",
                      flex: "1 1 30%",
                      display: "inline-flex",
                      alignItems: "center",
                      gap: 4,
                      justifyContent: "center",
                      background: mode === m.value ? "var(--forge-tint)" : undefined,
                      borderColor: mode === m.value ? "var(--trace)" : undefined,
                    }}
                    onClick={() => setMode(m.value)}
                  >
                    {m.icon}
                    {m.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Server model selector */}
            {(mode === "server" || mode === "both") && (
              <div style={{ marginTop: 12 }}>
                <label className="c-label">Server Model</label>
                <select
                  className="c-select"
                  value={selectedModelId}
                  onChange={(e) => setSelectedModelId(e.target.value)}
                >
                  <option value="">-- Select a server model --</option>
                  {serverModels.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name} ({m.operators.length} ops)
                    </option>
                  ))}
                </select>
                {serverModels.length === 0 && (
                  <p className="c-muted" style={{ fontSize: 11, marginTop: 4 }}>
                    No server models found. Upload one via the Models page.
                  </p>
                )}
              </div>
            )}

            <div
              ref={dropRef}
              className={`c-drop${dragActive ? " hover" : ""}`}
              role="button"
              tabIndex={0}
              aria-label="Drop an ONNX model file here, or press Enter to browse"
              onClick={() => fileInputRef.current?.click()}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  fileInputRef.current?.click();
                }
              }}
              onDragOver={(e) => {
                e.preventDefault();
                setDragActive(true);
              }}
              onDragLeave={() => setDragActive(false)}
              onDrop={onDrop}
              style={{ marginTop: 12, cursor: "pointer" }}
            >
              <Upload size={22} style={{ marginBottom: 8 }} />
              <div style={{ fontSize: 14, fontWeight: 600, color: "var(--ink)" }}>
                Drop <span className="mono">.onnx</span> model here
              </div>
              <div className="c-muted" style={{ fontSize: 12, marginTop: 4 }}>
                Loaded:{" "}
                <span className="mono" style={{ color: "var(--trace)" }}>
                  {modelName}
                </span>
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept=".onnx"
                style={{ display: "none" }}
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) handleModelFile(f);
                  e.target.value = "";
                }}
              />
            </div>

            <div style={{ marginTop: 18 }}>
              <label className="c-label">Input Shape</label>
              <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                <span className="mono" style={{ color: "var(--ink-muted)" }}>
                  [
                </span>
                {shape.map((d, i) => (
                  <input
                    key={i}
                    className="c-input"
                    type="number"
                    value={d}
                    onChange={(e) => updateShape(i, Number(e.target.value))}
                    style={{ width: 72, textAlign: "center" }}
                  />
                ))}
                <span className="mono" style={{ color: "var(--ink-muted)" }}>
                  ]
                </span>
                <button
                  className="c-btn c-btn-ghost"
                  onClick={() => setShape((s) => [...s, 1])}
                  style={{ padding: "6px 10px" }}
                >
                  + dim
                </button>
                {shape.length > 1 && (
                  <button
                    className="c-btn c-btn-ghost"
                    onClick={() => setShape((s) => s.slice(0, -1))}
                    style={{ padding: "6px 10px" }}
                  >
                    − dim
                  </button>
                )}
              </div>
              <p className="c-muted" style={{ fontSize: 12, marginTop: 6 }}>
                Expecting{" "}
                <span className="mono" style={{ color: "var(--trace)" }}>
                  {expectedSize}
                </span>{" "}
                values.
              </p>
            </div>

            <div style={{ marginTop: 14 }}>
              <label className="c-label">Sample Input</label>
              <div style={{ display: "flex", gap: 6, marginBottom: 8 }}>
                {[
                  { value: "zeros", label: "Zeros" },
                  { value: "random", label: "Random" },
                  { value: "custom", label: "Custom" },
                ].map((s) => (
                  <button
                    key={s.value}
                    className={`c-preset ${sampleMode === s.value ? "active" : ""}`}
                    style={{
                      width: "auto",
                      padding: "5px 12px",
                      background: sampleMode === s.value ? "var(--forge-tint)" : undefined,
                      borderColor: sampleMode === s.value ? "var(--trace)" : undefined,
                    }}
                    onClick={() => applySample(s.value as SampleMode)}
                  >
                    {s.label}
                  </button>
                ))}
              </div>
              <textarea
                className="c-textarea"
                value={values}
                onChange={(e) => {
                  setValues(e.target.value);
                  setSampleMode("custom");
                }}
                placeholder="Comma or whitespace-separated numbers"
                rows={3}
              />
              <p
                className="c-muted"
                style={{ fontSize: 12, marginTop: 6, color: valid ? "var(--ok)" : "var(--risk)" }}
              >
                {valid
                  ? `✓ ${parsedValues.length} values parsed`
                  : `✗ Got ${parsedValues.length} — need ${expectedSize}`}
              </p>
            </div>

            <button
              className="c-btn c-btn-primary c-btn-full"
              onClick={run}
              disabled={!valid || running}
              style={{ marginTop: 18 }}
            >
              <Play size={15} /> {running ? "Running..." : "Run Inference"}
            </button>
            {running && <div className="c-loading-bar" style={{ marginTop: 12 }} />}

            {error && (
              <div
                role="alert"
                className="c-fade-in"
                style={{ marginTop: 12, color: "var(--risk)", fontSize: 13 }}
              >
                {error}
              </div>
            )}

            {result && (
              <div className="c-fade-in" style={{ marginTop: 22 }}>
                <div className="c-label">Output Tensor</div>
                <div className="c-grid-3" style={{ gap: 10, marginTop: 8 }}>
                  <div className="c-metric" style={{ padding: 12 }}>
                    <div className="c-metric-label">Latency</div>
                    <div className="c-metric-value" style={{ fontSize: 18 }}>
                      {result.latencyMs} ms
                    </div>
                  </div>
                  <div className="c-metric" style={{ padding: 12 }}>
                    <div className="c-metric-label">Shape</div>
                    <div className="c-metric-value" style={{ fontSize: 18 }}>
                      [{result.shape.join(", ")}]
                    </div>
                  </div>
                  <div className="c-metric" style={{ padding: 12 }}>
                    <div className="c-metric-label">Engine</div>
                    <div className="c-metric-value" style={{ fontSize: 14, color: "var(--trace)" }}>
                      {result.engine}
                    </div>
                  </div>
                </div>
                <pre className="c-code" style={{ marginTop: 12 }}>
                  {`{
  `}
                  <span className="s">"output"</span>
                  {`: [`}
                  {result.output.map((v) => (
                    <span key={v} className="n">
                      {v}
                    </span>
                  ))}
                  {`],
  `}
                  <span className="s">"shape"</span>
                  {`: [`}
                  <span className="n">{result.shape.join(", ")}</span>
                  {`],
  `}
                  <span className="s">"backend"</span>
                  {`: `}
                  <span className="s">"{result.engine}"</span>
                  {`
}`}
                </pre>
              </div>
            )}
          </div>

          {/* RIGHT — graph inspector */}
          <div className="c-card">
            <h3 className="c-h3">Model Graph Inspector</h3>
            <div className="c-grid-2" style={{ gap: 10, marginTop: 12 }}>
              <div className="c-metric" style={{ padding: 12 }}>
                <div className="c-metric-label">Model</div>
                <div className="mono" style={{ fontSize: 13, marginTop: 6, color: "var(--trace)" }}>
                  {modelName}
                </div>
              </div>
              <div className="c-metric" style={{ padding: 12 }}>
                <div className="c-metric-label">Producer</div>
                <div className="mono" style={{ fontSize: 13, marginTop: 6, color: "var(--trace)" }}>
                  {DEFAULT_MODEL.producer}
                </div>
              </div>
              <div className="c-metric" style={{ padding: 12 }}>
                <div className="c-metric-label">IR Version</div>
                <div className="c-metric-value" style={{ fontSize: 20 }}>
                  {DEFAULT_MODEL.irVersion}
                </div>
              </div>
              <div className="c-metric" style={{ padding: 12 }}>
                <div className="c-metric-label">Opset</div>
                <div className="c-metric-value" style={{ fontSize: 20 }}>
                  {DEFAULT_MODEL.opset}
                </div>
              </div>
            </div>

            <div className="c-label" style={{ marginTop: 20 }}>
              Graph Nodes ({nodes.length})
            </div>
            <div style={{ display: "grid", gap: 6, marginTop: 8 }}>
              {nodes.map((n, i) => (
                <div key={i}>
                  <button
                    className="c-node"
                    style={{ width: "100%" }}
                    onClick={() => setExpanded(expanded === i ? null : i)}
                    aria-expanded={expanded === i}
                    aria-controls={`node-detail-${i}`}
                  >
                    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                      {expanded === i ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                      <span style={{ color: "var(--ink-muted)", width: 22 }} className="mono">
                        #{i}
                      </span>
                      <span className="c-node-op">{n.op}</span>
                    </div>
                    <span className="c-node-shape">→ {n.output}</span>
                  </button>
                  {expanded === i && (
                    <div id={`node-detail-${i}`} className="c-node-details c-fade-in">
                      <div>
                        <span style={{ color: "var(--ink-muted)" }}>inputs: </span>
                        <span className="mono" style={{ color: "var(--trace)" }}>
                          {n.inputs}
                        </span>
                      </div>
                      <div>
                        <span style={{ color: "var(--ink-muted)" }}>output: </span>
                        <span className="mono" style={{ color: "var(--ok)" }}>
                          {n.output}
                        </span>
                      </div>
                      {n.attrs &&
                        Object.entries(n.attrs).map(([k, v]) => (
                          <div key={k}>
                            <span style={{ color: "var(--ink-muted)" }}>{k}: </span>
                            <span className="mono" style={{ color: "var(--warn)" }}>
                              {v}
                            </span>
                          </div>
                        ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      <style>{`
        .c-history-item:hover {
          background: var(--paper-2);
        }
      `}</style>
    </CrucibleLayout>
  );
}
