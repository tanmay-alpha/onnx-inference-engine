import { createFileRoute, Link } from "@tanstack/react-router";
import { CrucibleLayout } from "../components/crucible/Layout";

export const Route = createFileRoute("/story")({
  head: () => ({
    meta: [
      { title: "Story · Crucible" },
      { name: "description", content: "Why Crucible exists: ONNX Runtime is 50MB+, PyTorch is 750MB+, and neither fits in a browser tab. So I wrote it from scratch." },
      { property: "og:title", content: "Crucible — Why This Exists" },
      { property: "og:description", content: "A first-person build story: hand-writing an ONNX protobuf decoder and keeping a C++ engine and a Rust engine bit-for-bit identical." },
      { property: "og:url", content: "/story" },
    ],
    links: [{ rel: "canonical", href: "/story" }],
  }),
  component: StoryPage,
});

function StoryPage() {
  return (
    <CrucibleLayout>
      <section className="c-container" style={{ maxWidth: 780, marginLeft: "auto", marginRight: "auto" }}>
        <span className="c-badge c-badge-info">Story</span>
        <h1 className="c-h2" style={{ fontSize: 44, marginTop: 14 }}>
          I wanted to run a fraud model inside a browser tab.
        </h1>
        <p className="c-muted" style={{ fontSize: 16, marginTop: 12 }}>
          The existing runtimes couldn't. So I wrote a new one — twice.
        </p>

        <div className="c-divider" style={{ margin: "32px 0" }} />

        <p style={{ fontSize: 17, lineHeight: 1.7, color: "var(--ink)" }}>
          The prompt was small and specific. Score a transaction as fraud or not-fraud, on the client,
          without shipping the features to a server. Client-side, because the demo brief said "no
          server calls," and because sending raw account balances across the wire to a hosted inference
          endpoint has real consequences under PCI-DSS.
        </p>

        <p style={{ fontSize: 17, lineHeight: 1.7, color: "var(--ink)", marginTop: 18 }}>
          I tried the obvious thing first: use ONNX Runtime Web. The distributable is over 50 MB
          before I've loaded a single model. PyTorch's mobile export is 750 MB+ on disk before you
          strip anything out. Neither is a serious answer for a fraud check that has to fit on a
          banking landing page. So I stopped looking for a runtime to wrap and started writing one.
        </p>

        <div style={{ margin: "36px 0", padding: "22px 26px", background: "var(--paper)", border: "1px solid var(--rule)", borderLeft: "3px solid var(--trace)" }}>
          <div className="c-eyebrow" style={{ marginBottom: 8 }}>Constraint</div>
          <div style={{ fontSize: 18, color: "var(--ink)", lineHeight: 1.5 }}>
            A runtime the browser will actually download. Under 5 MB compressed. No native
            dependencies at load time. Deterministic output.
          </div>
        </div>

        <h2 className="c-h2" style={{ fontSize: 26, marginTop: 40, marginBottom: 12 }}>Two hard problems that don't show on a benchmark chart.</h2>

        <h3 className="c-h3" style={{ marginTop: 24, fontSize: 17 }}>Hand-writing the protobuf decoder.</h3>
        <p style={{ fontSize: 16, lineHeight: 1.7, color: "var(--ink-default)", marginTop: 8 }}>
          <span className="mono">.onnx</span> files are protobuf-encoded. The obvious path is to
          run <span className="mono">protoc</span> against the ONNX <span className="mono">.proto</span>{" "}
          schema and get generated C++ or Rust readers for free. I didn't want that dependency in the
          build. So I wrote a decoder from the wire-format spec: read a tag byte, split it into a
          field number and a wire type, dispatch on the wire type — varint, 64-bit, length-delimited,
          32-bit — and repeat. The whole reader is a few hundred lines. It understands only the
          fields Crucible actually uses; unknown fields are skipped correctly, which is what the spec
          asks for.
        </p>

        <h3 className="c-h3" style={{ marginTop: 24, fontSize: 17 }}>Keeping two implementations identical.</h3>
        <p style={{ fontSize: 16, lineHeight: 1.7, color: "var(--ink-default)", marginTop: 8 }}>
          The C++ core uses Eigen for its matmul path. The Rust build for the browser can't rely on
          Eigen — it's a pure-Rust kernel set that lowers to <span className="mono">wasm-simd128</span>.
          Two implementations means two chances for the numbers to drift. I keep them honest with
          fixed-seed golden tests: the fraud model gets a fixed input tensor, both implementations
          execute the graph, and every intermediate tensor is compared bit-for-bit at the
          representation level for integer tensors and inside a tight epsilon for floats. When the
          test fails I know before I ship.
        </p>

        <div style={{ margin: "40px 0", padding: "24px 26px", background: "var(--paper-2)", border: "1px solid var(--rule)" }}>
          <div className="c-eyebrow" style={{ marginBottom: 12 }}>The numbers I care about</div>
          <div className="c-grid-3" style={{ gap: 12 }}>
            <div>
              <div className="mono" style={{ fontSize: 22, color: "var(--ink)", letterSpacing: "-.02em" }}>3.1 MB</div>
              <div className="c-metric-sub">runtime binary (wasm)</div>
            </div>
            <div>
              <div className="mono" style={{ fontSize: 22, color: "var(--ink)", letterSpacing: "-.02em" }}>15</div>
              <div className="c-metric-sub">operators implemented</div>
            </div>
            <div>
              <div className="mono" style={{ fontSize: 22, color: "var(--ink)", letterSpacing: "-.02em" }}>1.18 ms</div>
              <div className="c-metric-sub">fraud inference, warm</div>
            </div>
          </div>
        </div>

        <h2 className="c-h2" style={{ fontSize: 26, marginTop: 32, marginBottom: 12 }}>What I'd do next.</h2>
        <p style={{ fontSize: 16, lineHeight: 1.7, color: "var(--ink-default)" }}>
          Int8 quantization on the MatMul path, so the fraud model shrinks another 3–4×. A parity
          harness in CI so the C++/Rust check runs on every commit, not just when I remember.
          Biometric matching next — the same "features never leave the tab" argument holds even
          harder for face embeddings.
        </p>

        <div style={{ marginTop: 36, display: "flex", gap: 12, flexWrap: "wrap" }}>
          <Link to="/architecture" className="c-btn c-btn-primary">See the architecture</Link>
          <Link to="/roadmap" className="c-btn c-btn-secondary">See the roadmap</Link>
        </div>
      </section>
    </CrucibleLayout>
  );
}
