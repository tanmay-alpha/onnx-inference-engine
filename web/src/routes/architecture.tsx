import { createFileRoute, Link } from "@tanstack/react-router";
import { CrucibleLayout } from "../components/crucible/Layout";
import { ArrowRight } from "lucide-react";

export const Route = createFileRoute("/architecture")({
  head: () => ({
    meta: [
      { title: "Architecture · Crucible" },
      {
        name: "description",
        content:
          "How Crucible loads an ONNX model, decodes its protobuf bytes, and schedules kernels — both in native C++ and in Rust/WASM.",
      },
      { property: "og:title", content: "Crucible — Architecture" },
      {
        property: "og:description",
        content:
          "Row-major NCHW tensors, Kahn's topological execution, and a hosted-vs-local data-path comparison.",
      },
      { property: "og:url", content: "/architecture" },
    ],
    links: [{ rel: "canonical", href: "/architecture" }],
  }),
  component: ArchitecturePage,
});

function Pipe({ from, to }: { from: string; to: string }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, color: "var(--ink-muted)" }}>
      <span className="mono" style={{ color: "var(--ink)" }}>
        {from}
      </span>
      <ArrowRight size={12} />
      <span className="mono" style={{ color: "var(--ink)" }}>
        {to}
      </span>
    </div>
  );
}

function ArchitecturePage() {
  return (
    <CrucibleLayout>
      <section className="c-container">
        <div style={{ maxWidth: 720, marginBottom: 40 }}>
          <span className="c-badge c-badge-info">Architecture</span>
          <h1 className="c-h2" style={{ fontSize: 42, marginTop: 14 }}>
            How Crucible runs a model.
          </h1>
          <p className="c-muted">
            Two implementations of the same engine — native C++17 for benchmarks and embedded
            targets, and a Rust reimplementation compiled to WebAssembly for the browser. Same
            graph, same operator semantics, same numeric output on the fixed test set.
          </p>
        </div>

        {/* ---------- Pipeline ---------- */}
        <div className="c-rule left">
          <span className="num">01</span> · Request pipeline
        </div>
        <div className="c-plate" style={{ marginBottom: 32 }}>
          <svg
            viewBox="0 0 900 220"
            style={{ width: "100%", height: "auto" }}
            role="img"
            aria-label="Request pipeline"
          >
            {[
              { x: 20, label: "React page", sub: "src/app/fraud" },
              { x: 200, label: "crucible-wasm.ts", sub: "TS wrapper" },
              { x: 400, label: "WASM bindings", sub: "wasm-bindgen" },
              { x: 620, label: "Rust engine", sub: "kernels + scheduler" },
              { x: 810, label: "Score", sub: "f32 out" },
            ].map((s, i, arr) => (
              <g key={s.label}>
                <rect
                  x={s.x}
                  y={70}
                  width={140}
                  height={80}
                  fill="var(--paper)"
                  stroke="var(--ink)"
                />
                <text
                  x={s.x + 70}
                  y={104}
                  textAnchor="middle"
                  fontFamily="Inter Tight, sans-serif"
                  fontSize="13"
                  fontWeight="600"
                  fill="var(--ink)"
                >
                  {s.label}
                </text>
                <text
                  x={s.x + 70}
                  y={122}
                  textAnchor="middle"
                  fontFamily="JetBrains Mono, monospace"
                  fontSize="10"
                  fill="var(--ink-muted)"
                >
                  {s.sub}
                </text>
                {i < arr.length - 1 && (
                  <>
                    <line
                      x1={s.x + 140}
                      y1={110}
                      x2={arr[i + 1].x}
                      y2={110}
                      stroke="var(--trace)"
                      strokeWidth="1.5"
                    />
                    <polygon
                      points={`${arr[i + 1].x - 6},106 ${arr[i + 1].x},110 ${arr[i + 1].x - 6},114`}
                      fill="var(--trace)"
                    />
                  </>
                )}
              </g>
            ))}
            <text
              x={20}
              y={40}
              fontFamily="JetBrains Mono, monospace"
              fontSize="10"
              letterSpacing="2"
              fill="var(--ink-muted)"
            >
              SAMPLE PATH · fraud inference
            </text>
            <line x1={20} y1={190} x2={880} y2={190} stroke="var(--rule)" strokeDasharray="3 3" />
            <text
              x={450}
              y={210}
              textAnchor="middle"
              fontFamily="JetBrains Mono, monospace"
              fontSize="10"
              fill="var(--ink-muted)"
            >
              no network hop between any of these boxes
            </text>
          </svg>
        </div>

        {/* ---------- NCHW + Kahn's ---------- */}
        <div className="c-grid-2" style={{ marginBottom: 32 }}>
          <div className="c-card">
            <h3 className="c-h3">Tensors are flat, row-major NCHW.</h3>
            <p className="c-muted" style={{ marginTop: 8 }}>
              A shape like <span className="mono">[N, C, H, W]</span> is one contiguous{" "}
              <span className="mono">float32</span>
              buffer. Strides for each axis are precomputed once —{" "}
              <span className="mono">stride[i] = shape[i+1] * stride[i+1]</span> — so an index like{" "}
              <span className="mono">(n, c, h, w)</span> lowers to a single multiply-add. It matters
              because every MatMul, every Conv, every bias broadcast is ultimately linear reads
              through this buffer, and the tighter the memory story the more the SIMD lanes stay
              full.
            </p>
            <pre className="c-code" style={{ marginTop: 12 }}>
              {`shape   = [1, 3, 224, 224]
stride  = [3*224*224, 224*224, 224, 1]
offset  = n*stride[0] + c*stride[1] + h*stride[2] + w*stride[3]`}
            </pre>
          </div>
          <div className="c-card">
            <h3 className="c-h3">Execution order is Kahn's algorithm.</h3>
            <p className="c-muted" style={{ marginTop: 8 }}>
              An ONNX graph is a DAG of operators. Crucible builds an{" "}
              <span className="mono">in-degree[node]</span> map, seeds a ready-set with every
              zero-in-degree node, and pops nodes off in topological order — decrementing
              successors, appending them when their in-degree hits zero. Deterministic,
              cycle-detecting, and small.
            </p>
            <pre className="c-code" style={{ marginTop: 12 }}>
              {`ready = { n : in_degree[n] == 0 }
while ready not empty:
  n = ready.pop()
  execute(n)
  for m in successors(n):
    if --in_degree[m] == 0: ready.add(m)`}
            </pre>
          </div>
        </div>

        {/* ---------- Data path comparison ---------- */}
        <div className="c-rule left">
          <span className="num">02</span> · Data path
        </div>
        <div className="c-two-col" style={{ marginBottom: 32 }}>
          <div className="c-plate">
            <div className="c-eyebrow" style={{ color: "var(--risk)" }}>
              with a hosted model API
            </div>
            <div style={{ marginTop: 14, display: "flex", flexDirection: "column", gap: 10 }}>
              <Pipe from="tab" to="TLS to your server" />
              <Pipe from="your server" to="TLS to model host" />
              <Pipe from="model host" to="raw features in memory" />
              <Pipe from="model host" to="score back to server" />
              <Pipe from="your server" to="score back to tab" />
            </div>
            <div className="c-divider" />
            <div style={{ fontSize: 13, color: "var(--risk)" }}>
              Raw transaction features cross three trust boundaries. The model host becomes an
              in-scope processor for PCI-DSS / GDPR purposes.
            </div>
          </div>
          <div className="c-plate">
            <div className="c-eyebrow" style={{ color: "var(--ok)" }}>
              with crucible
            </div>
            <div style={{ marginTop: 14, display: "flex", flexDirection: "column", gap: 10 }}>
              <Pipe from="tab (once)" to="fetch .onnx (static CDN)" />
              <Pipe from="tab" to="wasm execute" />
              <Pipe from="tab" to="score (in same tab)" />
            </div>
            <div className="c-divider" />
            <div style={{ fontSize: 13, color: "var(--ok)" }}>
              Raw features never leave the tab. Only the final score is available to the app. The
              model host is a static file server — not a processor of feature data.
            </div>
          </div>
        </div>

        {/* ---------- Two builds ---------- */}
        <div className="c-rule left">
          <span className="num">03</span> · Two builds, one engine
        </div>
        <div className="c-spec">
          <div className="c-spec-row">
            <div className="k">native core</div>
            <div className="v">C++17 · Eigen for GEMM · im2col conv · pthreads optional</div>
          </div>
          <div className="c-spec-row">
            <div className="k">wasm build</div>
            <div className="v">
              Rust reimplementation · wasm-simd128 · no Eigen · wasm-bindgen glue
            </div>
          </div>
          <div className="c-spec-row">
            <div className="k">shared</div>
            <div className="v">
              Same ONNX decoder shape · same operator set · same fixed-seed golden tests
            </div>
          </div>
          <div className="c-spec-row">
            <div className="k">parity</div>
            <div className="v">
              Kernels compared bit-for-bit on the fraud model's calibration inputs
            </div>
          </div>
          <div className="c-spec-row">
            <div className="k">binary size</div>
            <div className="v">Native ≈ 1.4 MB · WASM ≈ 3.1 MB after wasm-opt --O3</div>
          </div>
        </div>

        <div style={{ marginTop: 32, display: "flex", gap: 10, flexWrap: "wrap" }}>
          <Link to="/playground" className="c-btn c-btn-primary">
            Inspect the graph <ArrowRight size={14} />
          </Link>
          <Link to="/benchmark" className="c-btn c-btn-secondary">
            See both builds benchmarked
          </Link>
        </div>
      </section>
    </CrucibleLayout>
  );
}
