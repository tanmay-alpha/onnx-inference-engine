import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { CrucibleLayout } from "../components/crucible/Layout";

export const Route = createFileRoute("/docs")({
  head: () => ({
    meta: [
      { title: "Operator Docs · Crucible" },
      { name: "description", content: "Reference documentation for every ONNX operator implemented in the Crucible engine." },
      { property: "og:title", content: "Crucible Operator Reference" },
      { property: "og:description", content: "Every kernel, attribute, and FFI signature." },
    ],
  }),
  component: DocsPage,
});

interface Op {
  name: string;
  signature: string;
  inputs: string;
  output: string;
  attrs?: { name: string; type: string; note: string }[];
  usage: string;
}

interface Category { key: string; label: string; ops: Op[] }

const CATEGORIES: Category[] = [
  { key: "math", label: "Math Ops", ops: [
    { name: "MatMul", signature: "MatMul(A: T[..., M, K], B: T[..., K, N]) → T[..., M, N]",
      inputs: "A, B tensors with broadcastable batch dims", output: "product tensor",
      usage: `// crucible_core/ops/matmul.hpp
Tensor y = crucible::matmul(a, b);
// Rust FFI
let y = crucible::matmul(&a, &b)?;` },
    { name: "Add", signature: "Add(A: T, B: T) → T",
      inputs: "broadcastable tensors A, B", output: "elementwise sum",
      usage: `Tensor y = crucible::add(a, b);` },
    { name: "Mul", signature: "Mul(A: T, B: T) → T",
      inputs: "broadcastable tensors A, B", output: "elementwise product",
      usage: `Tensor y = crucible::mul(a, b);` },
  ]},
  { key: "act", label: "Activations", ops: [
    { name: "Relu", signature: "Relu(X: T) → T",
      inputs: "X ∈ T[...]", output: "max(0, X)",
      usage: `Tensor y = crucible::relu(x); // SIMD-vectorized via Eigen` },
    { name: "Sigmoid", signature: "Sigmoid(X: T) → T",
      inputs: "X ∈ T[...]", output: "1 / (1 + exp(-X))",
      usage: `Tensor y = crucible::sigmoid(x);` },
    { name: "Softmax", signature: "Softmax(X: T, axis: int) → T",
      inputs: "X ∈ T[...], axis of reduction", output: "normalized distribution",
      attrs: [{ name: "axis", type: "int", note: "reduction axis, default -1" }],
      usage: `Tensor y = crucible::softmax(x, /*axis=*/-1);` },
  ]},
  { key: "conv", label: "Convolution", ops: [
    { name: "Conv2D", signature: "Conv2D(X: T[N,C,H,W], W: T[K,C,kH,kW], B?: T[K]) → T[N,K,Hout,Wout]",
      inputs: "activation X, kernel W, optional bias B", output: "convolved feature map",
      attrs: [
        { name: "stride",  type: "int[2]", note: "default [1,1]" },
        { name: "padding", type: "int[4]", note: "top, left, bottom, right" },
        { name: "dilation",type: "int[2]", note: "default [1,1]" },
        { name: "groups",  type: "int",    note: "default 1" },
      ],
      usage: `ConvOpts o{.stride={1,1}, .padding={1,1,1,1}};
Tensor y = crucible::conv2d(x, w, b, o);` },
  ]},
  { key: "norm", label: "Normalization", ops: [
    { name: "BatchNorm", signature: "BatchNorm(X, scale, B, mean, var, eps) → T",
      inputs: "activation X and 4 param tensors", output: "normalized activation",
      attrs: [{ name: "epsilon", type: "float", note: "default 1e-5" }],
      usage: `Tensor y = crucible::batchnorm(x, scale, bias, mean, var, 1e-5f);` },
    { name: "LayerNorm", signature: "LayerNorm(X, scale, B, axis, eps) → T",
      inputs: "activation X, learnable scale + bias", output: "normalized activation",
      attrs: [
        { name: "axis",    type: "int",   note: "default -1" },
        { name: "epsilon", type: "float", note: "default 1e-5" },
      ],
      usage: `Tensor y = crucible::layernorm(x, scale, bias, -1, 1e-5f);` },
  ]},
  { key: "pool", label: "Pooling", ops: [
    { name: "MaxPool2D", signature: "MaxPool2D(X: T[N,C,H,W]) → T[N,C,Hout,Wout]",
      inputs: "activation X", output: "max-pooled tensor",
      attrs: [
        { name: "kernel",  type: "int[2]", note: "e.g. [2,2]" },
        { name: "stride",  type: "int[2]", note: "default = kernel" },
        { name: "padding", type: "int[4]", note: "default [0,0,0,0]" },
      ],
      usage: `Tensor y = crucible::maxpool2d(x, {2,2}, {2,2});` },
    { name: "AvgPool2D", signature: "AvgPool2D(X: T[N,C,H,W]) → T[N,C,Hout,Wout]",
      inputs: "activation X", output: "average-pooled tensor",
      usage: `Tensor y = crucible::avgpool2d(x, {2,2}, {2,2});` },
    { name: "GlobalAvgPool", signature: "GlobalAvgPool(X: T[N,C,H,W]) → T[N,C,1,1]",
      inputs: "activation X", output: "spatially-collapsed tensor",
      usage: `Tensor y = crucible::global_avg_pool(x);` },
  ]},
  { key: "shape", label: "Shapes", ops: [
    { name: "Reshape", signature: "Reshape(X: T, shape: int[]) → T[shape]",
      inputs: "X and target shape (at most one -1)", output: "view over same data",
      usage: `Tensor y = crucible::reshape(x, {1, -1});` },
    { name: "Transpose", signature: "Transpose(X: T, perm: int[]) → T",
      inputs: "X and permutation", output: "axis-permuted tensor",
      attrs: [{ name: "perm", type: "int[]", note: "axis permutation" }],
      usage: `Tensor y = crucible::transpose(x, {0, 2, 1});` },
    { name: "Concat", signature: "Concat(inputs: T[]*, axis: int) → T",
      inputs: "N tensors sharing all dims except axis", output: "concatenated tensor",
      attrs: [{ name: "axis", type: "int", note: "concatenation axis" }],
      usage: `Tensor y = crucible::concat({a, b, c}, /*axis=*/1);` },
  ]},
];

function DocsPage() {
  const [active, setActive] = useState(CATEGORIES[0].key);

  const scrollTo = (key: string) => {
    setActive(key);
    const el = document.getElementById(`op-cat-${key}`);
    if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  return (
    <CrucibleLayout>
      <section className="c-container">
        <div style={{ marginBottom: 32, maxWidth: 720 }}>
          <span className="c-badge c-badge-info">Reference</span>
          <h1 className="c-h2" style={{ fontSize: 44, marginTop: 14 }}>API &amp; Operator Docs</h1>
          <p className="c-muted">Every operator implemented in the Crucible core, with C++ signatures and FFI examples.</p>
        </div>

        <div className="c-docs">
          <aside className="c-sidebar">
            <div className="c-label" style={{ marginBottom: 10 }}>Categories</div>
            {CATEGORIES.map((c) => (
              <button key={c.key} className={`c-sidebar-item${active === c.key ? " active" : ""}`} onClick={() => scrollTo(c.key)}>
                {c.label} <span style={{ color: "var(--ink-muted)", marginLeft: 6 }}>({c.ops.length})</span>
              </button>
            ))}
          </aside>

          <div>
            {CATEGORIES.map((c) => (
              <div key={c.key} id={`op-cat-${c.key}`} style={{ scrollMarginTop: 100, marginBottom: 40 }}>
                <h2 className="c-h2" style={{ fontSize: 24, marginBottom: 16 }}>{c.label}</h2>
                {c.ops.map((op) => (
                  <div key={op.name} className="c-op-card">
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 8 }}>
                      <div>
                        <div className="c-op-name">{op.name}</div>
                        <div className="c-op-sig">{op.signature}</div>
                      </div>
                      <span className="c-badge">{c.label}</span>
                    </div>

                    <div className="c-divider" style={{ margin: "16px 0" }} />

                    <div className="c-grid-2">
                      <div>
                        <div className="c-label">Inputs</div>
                        <div className="c-muted mono" style={{ marginTop: 4 }}>{op.inputs}</div>
                      </div>
                      <div>
                        <div className="c-label">Output</div>
                        <div className="c-muted mono" style={{ marginTop: 4 }}>{op.output}</div>
                      </div>
                    </div>

                    {op.attrs && op.attrs.length > 0 && (
                      <div style={{ marginTop: 14 }}>
                        <div className="c-label">Attributes</div>
                        <table className="c-table" style={{ marginTop: 4 }}>
                          <tbody>
                            {op.attrs.map((a) => (
                              <tr key={a.name}>
                                <td style={{ color: "var(--trace)", width: "20%" }}>{a.name}</td>
                                <td style={{ color: "var(--warn)", width: "20%" }}>{a.type}</td>
                                <td style={{ color: "var(--ink-muted)", fontFamily: "inherit" }}>{a.note}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}

                    <div style={{ marginTop: 14 }}>
                      <div className="c-label">Usage</div>
                      <pre className="c-code" style={{ marginTop: 6 }}>{op.usage}</pre>
                    </div>
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>
      </section>
    </CrucibleLayout>
  );
}
