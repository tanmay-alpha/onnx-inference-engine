import { createFileRoute, Link } from "@tanstack/react-router";
import { CrucibleLayout } from "../components/crucible/Layout";

export const Route = createFileRoute("/story")({
  head: () => ({
    meta: [
      { title: "Story · Crucible" },
      {
        name: "description",
        content:
          "Why Crucible exists: a from-scratch ONNX engine in C++17, plus a small Rust/WASM runtime so a fraud model can be scored inside a browser tab.",
      },
      { property: "og:title", content: "Crucible — Why This Exists" },
      {
        property: "og:description",
        content:
          "A first-person build story: hand-writing an ONNX protobuf decoder, a C++ engine checked against ONNX Runtime, and a tiny WASM runtime for the browser.",
      },
    ],
    links: [{ rel: "canonical", href: "/story" }],
  }),
  component: StoryPage,
});

function StoryPage() {
  return (
    <CrucibleLayout>
      <section
        className="c-container"
        style={{ maxWidth: 780, marginLeft: "auto", marginRight: "auto" }}
      >
        <span className="c-badge c-badge-info">Story</span>
        <h1 className="c-h2" style={{ fontSize: 44, marginTop: 14 }}>
          I wanted to run a fraud model inside a browser tab.
        </h1>
        <p className="c-muted" style={{ fontSize: 16, marginTop: 12 }}>
          So I wrote an ONNX runtime from scratch — a C++ engine, and a tiny Rust one for the tab.
        </p>

        <div className="c-divider" style={{ margin: "32px 0" }} />

        <p style={{ fontSize: 17, lineHeight: 1.7, color: "var(--ink)" }}>
          The prompt was small and specific. Score a transaction as fraud or not-fraud, on the
          client, without shipping the features to a server. Client-side, because the demo brief
          said "no server calls," and because sending raw account balances across the wire to a
          hosted inference endpoint has real consequences under PCI-DSS.
        </p>

        <p style={{ fontSize: 17, lineHeight: 1.7, color: "var(--ink)", marginTop: 18 }}>
          The obvious answer is to wrap an existing runtime. I wanted to understand what a runtime
          actually does — parse the graph, order it, run the kernels — so I wrote one instead, and
          kept the browser part as small as the fraud model allows.
        </p>

        <div
          style={{
            margin: "36px 0",
            padding: "22px 26px",
            background: "var(--paper)",
            border: "1px solid var(--rule)",
            borderLeft: "3px solid var(--trace)",
          }}
        >
          <div className="c-eyebrow" style={{ marginBottom: 8 }}>
            Constraint
          </div>
          <div style={{ fontSize: 18, color: "var(--ink)", lineHeight: 1.5 }}>
            A runtime small enough to ship on a landing page. No native dependencies at load time.
            Every result is either correct or an explicit error.
          </div>
        </div>

        <h2 className="c-h2" style={{ fontSize: 26, marginTop: 40, marginBottom: 12 }}>
          Two hard problems that don't show on a benchmark chart.
        </h2>

        <h3 className="c-h3" style={{ marginTop: 24, fontSize: 17 }}>
          Hand-writing the protobuf decoder.
        </h3>
        <p style={{ fontSize: 16, lineHeight: 1.7, color: "var(--ink-default)", marginTop: 8 }}>
          <span className="mono">.onnx</span> files are protobuf-encoded. The obvious path is to run{" "}
          <span className="mono">protoc</span> against the ONNX <span className="mono">.proto</span>{" "}
          schema and get generated C++ or Rust readers for free. I didn't want that dependency in
          the build. So I wrote a decoder from the wire-format spec: read a tag byte, split it into
          a field number and a wire type, dispatch on the wire type — varint, 64-bit,
          length-delimited, 32-bit — and repeat. The whole reader is a few hundred lines. It
          understands only the fields Crucible actually uses; unknown fields are skipped correctly,
          which is what the spec asks for.
        </p>

        <h3 className="c-h3" style={{ marginTop: 24, fontSize: 17 }}>
          Two runtimes, honestly scoped.
        </h3>
        <p style={{ fontSize: 16, lineHeight: 1.7, color: "var(--ink-default)", marginTop: 8 }}>
          The C++ core uses Eigen for its matrix products and im2col convolutions and implements 20
          ONNX operators — enough for ResNet18 and MobileNetV2. Its outputs are checked against ONNX
          Runtime in a 70-case parity suite. It is not faster than ONNX Runtime: single-threaded it
          is several times slower on those CNNs, and the benchmark page shows the measured gap. The
          browser runtime is a separate, much smaller Rust interpreter with 5 operators (MatMul,
          Add, Relu, Sigmoid, Softmax): it runs the fraud model, not CNNs, and rejects anything else
          with a clear error.
        </p>

        <div
          style={{
            margin: "40px 0",
            padding: "24px 26px",
            background: "var(--paper-2)",
            border: "1px solid var(--rule)",
          }}
        >
          <div className="c-eyebrow" style={{ marginBottom: 12 }}>
            The numbers I care about
          </div>
          <div className="c-grid-3" style={{ gap: 12 }}>
            <div>
              <div
                className="mono"
                style={{ fontSize: 22, color: "var(--ink)", letterSpacing: "-.02em" }}
              >
                &lt;100 KB
              </div>
              <div className="c-metric-sub">WASM runtime binary</div>
            </div>
            <div>
              <div
                className="mono"
                style={{ fontSize: 22, color: "var(--ink)", letterSpacing: "-.02em" }}
              >
                20 / 5
              </div>
              <div className="c-metric-sub">operators: native C++ / WASM</div>
            </div>
            <div>
              <div
                className="mono"
                style={{ fontSize: 22, color: "var(--ink)", letterSpacing: "-.02em" }}
              >
                70
              </div>
              <div className="c-metric-sub">parity cases vs ONNX Runtime</div>
            </div>
          </div>
        </div>

        <h2 className="c-h2" style={{ fontSize: 26, marginTop: 32, marginBottom: 12 }}>
          What I'd do next.
        </h2>
        <p style={{ fontSize: 16, lineHeight: 1.7, color: "var(--ink-default)" }}>
          Close the native performance gap with ONNX Runtime (threading, kernel fusion, faster
          convolution paths). Grow the WASM runtime past five operators so more than the fraud model
          runs in the browser, and add a parity check between the two runtimes on the models they
          share.
        </p>

        <div style={{ marginTop: 36, display: "flex", gap: 12, flexWrap: "wrap" }}>
          <Link to="/architecture" className="c-btn c-btn-primary">
            See the architecture
          </Link>
          <Link to="/roadmap" className="c-btn c-btn-secondary">
            See the roadmap
          </Link>
        </div>
      </section>
    </CrucibleLayout>
  );
}
