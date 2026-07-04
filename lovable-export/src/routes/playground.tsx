import { createFileRoute } from "@tanstack/react-router";
import { useMemo, useRef, useState } from "react";
import { Play, Upload, ChevronRight, ChevronDown } from "lucide-react";
import { CrucibleLayout } from "../components/crucible/Layout";

export const Route = createFileRoute("/playground")({
  head: () => ({
    meta: [
      { title: "Playground · Crucible" },
      { name: "description", content: "Drop an ONNX model, feed a tensor, inspect the graph — all in the browser." },
      { property: "og:title", content: "Crucible WASM Playground" },
      { property: "og:description", content: "Interactive ONNX runtime running entirely client-side." },
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

function PlaygroundPage() {
  const [modelName, setModelName] = useState(DEFAULT_MODEL.name);
  const [shape, setShape] = useState<number[]>([1, 7]);
  const [values, setValues] = useState("0.31, 0.55, 0.02, 1.0, 0.88, 0.12, 0.44");
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<{ latencyMs: number; output: number[]; shape: number[] } | null>(null);
  const [expanded, setExpanded] = useState<number | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const dropRef = useRef<HTMLDivElement>(null);

  const nodes = DEFAULT_MODEL.nodes;

  const parsedValues = useMemo(() => values.split(/[\s,]+/).filter(Boolean).map(Number), [values]);
  const expectedSize = shape.reduce((a, b) => a * b, 1);
  const valid = parsedValues.length === expectedSize && parsedValues.every((v) => !Number.isNaN(v));

  const run = () => {
    if (!valid) return;
    setRunning(true);
    setResult(null);
    setTimeout(() => {
      // Fake forward pass: sigmoid of weighted sum
      const sum = parsedValues.reduce((a, b, i) => a + b * (0.1 + i * 0.07), 0);
      const out = 1 / (1 + Math.exp(-sum + 1.5));
      setResult({
        latencyMs: Number((0.6 + Math.random() * 0.6).toFixed(2)),
        output: [Number(out.toFixed(6))],
        shape: [1, 1],
      });
      setRunning(false);
    }, 820);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    const f = e.dataTransfer.files[0];
    if (f) setModelName(f.name);
  };

  const updateShape = (i: number, v: number) => {
    setShape((s) => s.map((x, idx) => (idx === i ? Math.max(1, v) : x)));
  };

  return (
    <CrucibleLayout>
      <section className="c-container">
        <div style={{ marginBottom: 32, maxWidth: 720 }}>
          <span className="c-badge c-badge-info">Developer Console</span>
          <h1 className="c-h2" style={{ fontSize: 44, marginTop: 14 }}>WASM Inference Playground</h1>
          <p className="c-muted">Drop an <span className="mono">.onnx</span> model, define input shape and tensor values, then execute a forward pass inside the browser sandbox.</p>
        </div>

        <div className="c-two-col">
          {/* LEFT — inference console */}
          <div className="c-card">
            <h3 className="c-h3">Inference Console</h3>

            <div
              ref={dropRef}
              className={`c-drop${dragActive ? " hover" : ""}`}
              onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
              onDragLeave={() => setDragActive(false)}
              onDrop={onDrop}
              style={{ marginTop: 12 }}
            >
              <Upload size={22} style={{ marginBottom: 8 }} />
              <div style={{ fontSize: 14, fontWeight: 600, color: "var(--ink)" }}>Drop <span className="mono">.onnx</span> model here</div>
              <div className="c-muted" style={{ fontSize: 12, marginTop: 4 }}>Loaded: <span className="mono" style={{ color: "var(--trace)" }}>{modelName}</span></div>
            </div>

            <div style={{ marginTop: 18 }}>
              <label className="c-label">Input Shape</label>
              <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <span className="mono" style={{ color: "var(--ink-muted)" }}>[</span>
                {shape.map((d, i) => (
                  <input key={i} className="c-input" type="number" value={d} onChange={(e) => updateShape(i, Number(e.target.value))} style={{ width: 80, textAlign: "center" }} />
                ))}
                <span className="mono" style={{ color: "var(--ink-muted)" }}>]</span>
                <button className="c-btn c-btn-ghost" onClick={() => setShape((s) => [...s, 1])} style={{ padding: "6px 10px" }}>+ dim</button>
                {shape.length > 1 && (
                  <button className="c-btn c-btn-ghost" onClick={() => setShape((s) => s.slice(0, -1))} style={{ padding: "6px 10px" }}>− dim</button>
                )}
              </div>
              <p className="c-muted" style={{ fontSize: 12, marginTop: 6 }}>Expecting <span className="mono" style={{ color: "var(--trace)" }}>{expectedSize}</span> values.</p>
            </div>

            <div style={{ marginTop: 14 }}>
              <label className="c-label">Input Tensor Values</label>
              <textarea className="c-textarea" value={values} onChange={(e) => setValues(e.target.value)} placeholder="Comma or whitespace-separated numbers" />
              <p className="c-muted" style={{ fontSize: 12, marginTop: 6, color: valid ? "var(--ok)" : "var(--risk)" }}>
                {valid ? `✓ ${parsedValues.length} values parsed` : `✗ Got ${parsedValues.length} — need ${expectedSize}`}
              </p>
            </div>

            <button
              className="c-btn c-btn-primary c-btn-full"
              onClick={run}
              disabled={!valid || running}
              style={{ marginTop: 18 }}
            >
              <Play size={15} /> {running ? "Running WASM..." : "Run Inference"}
            </button>
            {running && <div className="c-loading-bar" style={{ marginTop: 12 }} />}

            {result && (
              <div className="c-fade-in" style={{ marginTop: 22 }}>
                <div className="c-label">Output Tensor</div>
                <div className="c-grid-3" style={{ gap: 10, marginTop: 8 }}>
                  <div className="c-metric" style={{ padding: 12 }}>
                    <div className="c-metric-label">Latency</div>
                    <div className="c-metric-value" style={{ fontSize: 18 }}>{result.latencyMs} ms</div>
                  </div>
                  <div className="c-metric" style={{ padding: 12 }}>
                    <div className="c-metric-label">Shape</div>
                    <div className="c-metric-value" style={{ fontSize: 18 }}>[{result.shape.join(", ")}]</div>
                  </div>
                  <div className="c-metric" style={{ padding: 12 }}>
                    <div className="c-metric-label">Bytes Sent</div>
                    <div className="c-metric-value" style={{ fontSize: 18, color: "var(--ok)" }}>0</div>
                  </div>
                </div>
                <pre className="c-code" style={{ marginTop: 12 }}>
{`{
  `}<span className="s">"output"</span>{`: [`}
{result.output.map((v) => <span key={v} className="n">{v}</span>)}
{`],
  `}<span className="s">"shape"</span>{`: [`}<span className="n">{result.shape.join(", ")}</span>{`],
  `}<span className="s">"backend"</span>{`: `}<span className="s">"wasm-simd128"</span>{`
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
                <div className="mono" style={{ fontSize: 13, marginTop: 6, color: "var(--trace)" }}>{modelName}</div>
              </div>
              <div className="c-metric" style={{ padding: 12 }}>
                <div className="c-metric-label">Producer</div>
                <div className="mono" style={{ fontSize: 13, marginTop: 6, color: "var(--trace)" }}>{DEFAULT_MODEL.producer}</div>
              </div>
              <div className="c-metric" style={{ padding: 12 }}>
                <div className="c-metric-label">IR Version</div>
                <div className="c-metric-value" style={{ fontSize: 20 }}>{DEFAULT_MODEL.irVersion}</div>
              </div>
              <div className="c-metric" style={{ padding: 12 }}>
                <div className="c-metric-label">Opset</div>
                <div className="c-metric-value" style={{ fontSize: 20 }}>{DEFAULT_MODEL.opset}</div>
              </div>
            </div>

            <div className="c-label" style={{ marginTop: 20 }}>Graph Nodes ({nodes.length})</div>
            <div style={{ display: "grid", gap: 6, marginTop: 8 }}>
              {nodes.map((n, i) => (
                <div key={i}>
                  <button className="c-node" style={{ width: "100%" }} onClick={() => setExpanded(expanded === i ? null : i)}>
                    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                      {expanded === i ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                      <span style={{ color: "var(--ink-muted)", width: 22 }} className="mono">#{i}</span>
                      <span className="c-node-op">{n.op}</span>
                    </div>
                    <span className="c-node-shape">→ {n.output}</span>
                  </button>
                  {expanded === i && (
                    <div className="c-node-details c-fade-in">
                      <div><span style={{ color: "var(--ink-muted)" }}>inputs: </span><span className="mono" style={{ color: "var(--trace)" }}>{n.inputs}</span></div>
                      <div><span style={{ color: "var(--ink-muted)" }}>output: </span><span className="mono" style={{ color: "var(--ok)" }}>{n.output}</span></div>
                      {n.attrs && Object.entries(n.attrs).map(([k, v]) => (
                        <div key={k}><span style={{ color: "var(--ink-muted)" }}>{k}: </span><span className="mono" style={{ color: "var(--warn)" }}>{v}</span></div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>
    </CrucibleLayout>
  );
}
