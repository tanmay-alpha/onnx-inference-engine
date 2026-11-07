import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { CrucibleLayout } from "../components/crucible/Layout";

export const Route = createFileRoute("/docs")({
  head: () => ({
    meta: [
      { title: "Operator Docs · Crucible" },
      {
        name: "description",
        content:
          "The 20 ONNX operators implemented by the Crucible native engine, the 5 implemented by the WASM runtime, and how to call each interface.",
      },
      { property: "og:title", content: "Crucible Operator Reference" },
      {
        property: "og:description",
        content: "Supported operators, attribute limits, and the Python / HTTP / C interfaces.",
      },
    ],
  }),
  component: DocsPage,
});

interface Op {
  name: string;
  signature: string;
  notes: string;
  attrs?: { name: string; note: string }[];
  /** Also implemented by the Rust WASM runtime. */
  wasm?: boolean;
}

interface Category {
  key: string;
  label: string;
  ops: Op[];
}

const CATEGORIES: Category[] = [
  {
    key: "linear",
    label: "Linear algebra",
    ops: [
      {
        name: "MatMul",
        signature: "MatMul(A, B) → Y",
        notes: "Matrix product via Eigen. WASM runtime: 2-D operands only.",
        wasm: true,
      },
      {
        name: "Gemm",
        signature: "Gemm(A, B, C?) → alpha·A'·B' + beta·C",
        notes: "A and B must be rank 2; C is broadcast to (M, N).",
        attrs: [
          { name: "alpha, beta", note: "scales, default 1.0" },
          { name: "transA, transB", note: "transpose flags, default 0" },
        ],
      },
      {
        name: "Add",
        signature: "Add(A, B) → A + B",
        notes: "Numpy-style multidirectional broadcasting (both engines).",
        wasm: true,
      },
    ],
  },
  {
    key: "act",
    label: "Activations",
    ops: [
      { name: "Relu", signature: "Relu(X) → max(0, X)", notes: "Elementwise.", wasm: true },
      {
        name: "Sigmoid",
        signature: "Sigmoid(X) → 1 / (1 + e^-X)",
        notes: "Elementwise.",
        wasm: true,
      },
      { name: "Tanh", signature: "Tanh(X) → tanh(X)", notes: "Elementwise." },
      {
        name: "LeakyRelu",
        signature: "LeakyRelu(X) → X if X ≥ 0 else alpha·X",
        notes: "Elementwise.",
        attrs: [{ name: "alpha", note: "default 0.01" }],
      },
      {
        name: "Elu",
        signature: "Elu(X) → X if X > 0 else alpha·(e^X − 1)",
        notes: "Elementwise.",
        attrs: [{ name: "alpha", note: "default 1.0" }],
      },
      {
        name: "Gelu",
        signature: "Gelu(X) → X·Φ(X)",
        notes: "Exact (erf) or tanh approximation.",
        attrs: [{ name: "approximate", note: '"none" (default) or "tanh"' }],
      },
      {
        name: "Softmax",
        signature: "Softmax(X) → normalised along axis",
        notes:
          "Opset ≥ 13 normalises one axis (default −1); opset < 13 uses the legacy 2-D coercion (default axis 1). Same semantics in the WASM runtime.",
        attrs: [{ name: "axis", note: "any valid axis" }],
        wasm: true,
      },
    ],
  },
  {
    key: "conv",
    label: "Convolution & normalisation",
    ops: [
      {
        name: "Conv",
        signature: "Conv(X[N,C,H,W], W[K,C/g,kH,kW], B?) → Y[N,K,Ho,Wo]",
        notes:
          "2-D only. im2col + Eigen GEMM; 1×1 convolutions skip im2col; depthwise uses a direct loop.",
        attrs: [
          { name: "strides, pads", note: "pads must be [top, left, bottom, right]" },
          { name: "group", note: "≥ 1 (grouped and depthwise supported)" },
          { name: "dilations", note: "only 1 is supported" },
          { name: "auto_pad", note: "NOTSET or VALID only" },
        ],
      },
      {
        name: "BatchNormalization",
        signature: "BatchNormalization(X, scale, B, mean, var) → Y",
        notes: "Inference mode only.",
        attrs: [
          { name: "epsilon", note: "default 1e-5" },
          { name: "training_mode", note: "must be 0" },
        ],
      },
    ],
  },
  {
    key: "pool",
    label: "Pooling",
    ops: [
      {
        name: "MaxPool",
        signature: "MaxPool(X[N,C,H,W]) → Y",
        notes: "2-D windows; the optional Indices output is not supported.",
        attrs: [
          { name: "kernel_shape", note: "required" },
          { name: "strides, pads", note: "as Conv" },
          { name: "ceil_mode, storage_order", note: "must be 0" },
        ],
      },
      {
        name: "AveragePool",
        signature: "AveragePool(X[N,C,H,W]) → Y",
        notes: "2-D windows.",
        attrs: [
          { name: "kernel_shape", note: "required" },
          { name: "ceil_mode", note: "must be 0" },
        ],
      },
      {
        name: "GlobalAveragePool",
        signature: "GlobalAveragePool(X[N,C,H,W]) → Y[N,C,1,1]",
        notes: "Mean over the spatial dimensions.",
      },
    ],
  },
  {
    key: "shape",
    label: "Shape & plumbing",
    ops: [
      {
        name: "Reshape",
        signature: "Reshape(X, shape) → Y",
        notes: "Target shape from the second input; supports 0 (copy input dim) and one −1.",
        attrs: [{ name: "allowzero", note: "must be 0" }],
      },
      {
        name: "Flatten",
        signature: "Flatten(X) → Y[prod(d<axis), prod(d≥axis)]",
        notes: "Metadata-only reshape.",
        attrs: [{ name: "axis", note: "default 1" }],
      },
      {
        name: "Concat",
        signature: "Concat(X1, …, Xn) → Y",
        notes: "All inputs share rank and every dim except axis.",
        attrs: [{ name: "axis", note: "required" }],
      },
      {
        name: "Identity",
        signature: "Identity(X) → X",
        notes: "Pass-through.",
      },
      {
        name: "Dropout",
        signature: "Dropout(X) → X",
        notes: "Inference: identity. The optional mask output is not supported.",
      },
    ],
  },
];

const INTERFACES: { title: string; code: string }[] = [
  {
    title: "Python (pybind11 module crucible_py)",
    code: `import numpy as np, crucible_py
model = crucible_py.load_model("resnet18.onnx")
print(crucible_py.check_model(model))      # [] when every op/attribute is supported
x = np.random.rand(1, 3, 224, 224).astype(np.float32)
y = crucible_py.run(model, {"input": x})    # first output, float32 ndarray`,
  },
  {
    title: "HTTP (FastAPI server)",
    code: `curl -X POST $API/convert -H "Authorization: Bearer $TOKEN" \\
     -F model_file=@fraud_detector.onnx          # -> onnx_model_id, unsupported_reasons
curl -X POST $API/infer -H "Authorization: Bearer $TOKEN" \\
     -H "Content-Type: application/json" \\
     -d '{"model_id":"<id>","input":[0,0,0,0,0,0,0],"input_shape":[1,7]}'`,
  },
  {
    title: "C ABI (shared library crucible)",
    code: `CrucibleModel* m = crucible_load("model.onnx");
if (!m) { fprintf(stderr, "%s\\n", crucible_last_error()); return 1; }
int64_t shape[2] = {1, 7};
CrucibleTensorDesc in = { shape, 2, 7, features };
float* out; CrucibleTensorDesc out_desc;
CrucibleStatus s = crucible_run(m, &in, 1, &out, &out_desc, 1);
crucible_free(m);`,
  },
  {
    title: "Browser (WASM runtime, wasm-bindgen)",
    code: `import init, { runInference, modelInfo } from "/wasm/crucible_wasm.js";
await init();
const info = JSON.parse(modelInfo(bytes));   // supported, unsupported_reasons, inputs…
const y = runInference(bytes, new Float32Array(x), new Int32Array([1, 7]));`,
  },
];

const OP_COUNT = CATEGORIES.reduce((n, c) => n + c.ops.length, 0);
const WASM_OPS = CATEGORIES.flatMap((c) => c.ops.filter((o) => o.wasm).map((o) => o.name));

function DocsPage() {
  const [active, setActive] = useState("interfaces");

  const scrollTo = (key: string) => {
    setActive(key);
    const el = document.getElementById(`op-cat-${key}`);
    if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  return (
    <CrucibleLayout>
      <section className="c-container">
        <div style={{ marginBottom: 32, maxWidth: 760 }}>
          <span className="c-badge c-badge-info">Reference</span>
          <h1 className="c-h2" style={{ fontSize: 44, marginTop: 14 }}>
            API &amp; Operator Docs
          </h1>
          <p className="c-muted">
            The native C++ engine implements {OP_COUNT} ONNX operators for float32 tensors. The
            browser runtime is a separate, much smaller Rust interpreter implementing{" "}
            {WASM_OPS.length} of them ({WASM_OPS.join(", ")}) — enough for the fraud model, not for
            CNNs. Models using anything else are rejected with an explicit reason, never run
            approximately.
          </p>
        </div>

        <div className="c-docs">
          <aside className="c-sidebar">
            <div className="c-label" style={{ marginBottom: 10 }}>
              Sections
            </div>
            <button
              className={`c-sidebar-item${active === "interfaces" ? " active" : ""}`}
              onClick={() => scrollTo("interfaces")}
            >
              Interfaces
            </button>
            {CATEGORIES.map((c) => (
              <button
                key={c.key}
                className={`c-sidebar-item${active === c.key ? " active" : ""}`}
                onClick={() => scrollTo(c.key)}
              >
                {c.label}{" "}
                <span style={{ color: "var(--ink-muted)", marginLeft: 6 }}>({c.ops.length})</span>
              </button>
            ))}
          </aside>

          <div>
            <div id="op-cat-interfaces" style={{ scrollMarginTop: 100, marginBottom: 40 }}>
              <h2 className="c-h2" style={{ fontSize: 24, marginBottom: 16 }}>
                Interfaces
              </h2>
              {INTERFACES.map((i) => (
                <div key={i.title} className="c-op-card">
                  <div className="c-op-name" style={{ fontSize: 15 }}>
                    {i.title}
                  </div>
                  <pre className="c-code" style={{ marginTop: 10 }}>
                    {i.code}
                  </pre>
                </div>
              ))}
            </div>

            {CATEGORIES.map((c) => (
              <div
                key={c.key}
                id={`op-cat-${c.key}`}
                style={{ scrollMarginTop: 100, marginBottom: 40 }}
              >
                <h2 className="c-h2" style={{ fontSize: 24, marginBottom: 16 }}>
                  {c.label}
                </h2>
                {c.ops.map((op) => (
                  <div key={op.name} className="c-op-card">
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "flex-start",
                        flexWrap: "wrap",
                        gap: 8,
                      }}
                    >
                      <div>
                        <div className="c-op-name">{op.name}</div>
                        <div className="c-op-sig">{op.signature}</div>
                      </div>
                      <div style={{ display: "flex", gap: 6 }}>
                        <span className="c-badge">native</span>
                        {op.wasm && <span className="c-badge c-badge-forge">wasm</span>}
                      </div>
                    </div>

                    <p className="c-muted" style={{ marginTop: 12 }}>
                      {op.notes}
                    </p>

                    {op.attrs && op.attrs.length > 0 && (
                      <div style={{ marginTop: 14 }}>
                        <div className="c-label">Attributes</div>
                        <table className="c-table" style={{ marginTop: 4 }}>
                          <tbody>
                            {op.attrs.map((a) => (
                              <tr key={a.name}>
                                <td style={{ color: "var(--trace)", width: "30%" }}>{a.name}</td>
                                <td style={{ color: "var(--ink-muted)", fontFamily: "inherit" }}>
                                  {a.note}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
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
