import { createFileRoute, Link } from "@tanstack/react-router";
import { useMemo, useState, useCallback, useEffect, useRef } from "react";
import {
  Play,
  Upload,
  ChevronRight,
  ChevronDown,
  History,
  Server,
  Cpu,
  Layers,
  LogIn,
  CloudUpload,
} from "lucide-react";
import { CrucibleLayout } from "../components/crucible/Layout";
import {
  BackendOfflineNotice,
  NativeUnavailableNotice,
} from "../components/crucible/BackendStatus";
import {
  FRAUD_MODEL_URL,
  getFraudModelBytes,
  getWasmModelInfo,
  runWasmInference,
  type WasmModelInfo,
} from "../lib/crucible-wasm";
import { fetchModels, getModel, runInference, uploadModel, type RegisteredModel } from "../lib/api";
import { ELEM_TYPE_NAMES, parseOnnxModel, type OnnxModelMeta } from "../lib/onnx-meta";
import { useBackendHealth, useIsLoggedIn } from "../lib/use-backend-health";

export const Route = createFileRoute("/playground")({
  head: () => ({
    meta: [
      { title: "Playground · Crucible" },
      {
        name: "description",
        content:
          "Load an ONNX model, inspect its real graph, and run it with the in-browser WASM runtime or the native C++ engine on the server.",
      },
      { property: "og:title", content: "Crucible Playground" },
      {
        property: "og:description",
        content: "Run ONNX models in the browser (Rust/WASM) or on the native Crucible server.",
      },
    ],
  }),
  component: PlaygroundPage,
  validateSearch: (search: Record<string, unknown>): { modelId?: string } =>
    typeof search.modelId === "string" && search.modelId ? { modelId: search.modelId } : {},
});

type InferenceMode = "wasm" | "server" | "both";
type SampleMode = "zeros" | "random" | "custom";
type BackendKind = "wasm" | "server";

const BACKEND_LABEL: Record<BackendKind, string> = {
  wasm: "Crucible WASM (browser)",
  server: "Crucible native (server)",
};

interface RunOutcome {
  kind: BackendKind;
  ok: boolean;
  error?: string;
  output?: number[];
  outputShape?: number[];
  /** `backend` field reported by the runtime that produced the result. */
  backend?: string;
  /** WASM: runInference call time. Server: engine-only time reported by the API. */
  computeMs?: number;
  /** Server only: browser-measured fetch round trip (network + queueing + engine). */
  roundTripMs?: number;
}

interface HistoryEntry {
  id: string;
  timestamp: number;
  model: string;
  mode: InferenceMode;
  values: number[];
  shape: number[];
  outcomes: RunOutcome[];
}

const fmtShape = (s: number[] | null | undefined) =>
  s ? `[${s.map((d) => (d < 0 ? "?" : d)).join(", ")}]` : "unknown";

/** Browser timers are coarsened (~0.1 ms without cross-origin isolation). */
const fmtMs = (ms: number) => (ms === 0 ? "<0.1" : ms < 1 ? ms.toFixed(3) : ms.toFixed(2));

/** Concrete starting shape for a declared input: symbolic dims become 1. */
function concreteShape(declared: number[] | null | undefined): number[] {
  if (!declared || declared.length === 0) return [1];
  return declared.map((d) => (d > 0 ? d : 1));
}

function PlaygroundPage() {
  const { modelId: initialModelId } = Route.useSearch();
  const health = useBackendHealth();
  const loggedIn = useIsLoggedIn();

  // ---- local model (bytes + parsed metadata) ----
  const [modelName, setModelName] = useState("fraud_detector.onnx");
  const [modelBytes, setModelBytes] = useState<Uint8Array | null>(null);
  const [meta, setMeta] = useState<OnnxModelMeta | null>(null);
  const [wasmInfo, setWasmInfo] = useState<WasmModelInfo | null>(null);
  const [wasmInfoError, setWasmInfoError] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  // ---- input ----
  const [shape, setShape] = useState<number[]>([1]);
  const [values, setValues] = useState("");
  const [sampleMode, setSampleMode] = useState<SampleMode>("zeros");

  // ---- server ----
  const [mode, setMode] = useState<InferenceMode>(initialModelId ? "server" : "wasm");
  const [serverModels, setServerModels] = useState<RegisteredModel[]>([]);
  const [serverError, setServerError] = useState<string | null>(null);
  const [selectedModelId, setSelectedModelId] = useState(initialModelId ?? "");
  const [selectedServerModel, setSelectedServerModel] = useState<RegisteredModel | null>(null);
  const [uploading, setUploading] = useState(false);

  // ---- results ----
  const [running, setRunning] = useState(false);
  const [outcomes, setOutcomes] = useState<RunOutcome[]>([]);
  const [expanded, setExpanded] = useState<number | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const declaredInput = meta?.inputs[0] ?? null;

  /** Parse bytes, inspect with the WASM runtime, and reset the input to the declared shape. */
  const loadModel = useCallback(async (name: string, bytes: Uint8Array) => {
    setModelName(name);
    setOutcomes([]);
    setLoadError(null);
    setWasmInfo(null);
    setWasmInfoError(null);
    let parsed: OnnxModelMeta;
    try {
      parsed = parseOnnxModel(bytes);
    } catch (e) {
      setModelBytes(null);
      setMeta(null);
      setLoadError(`${name}: ${e instanceof Error ? e.message : String(e)}`);
      return;
    }
    setModelBytes(bytes);
    setMeta(parsed);
    const s = concreteShape(parsed.inputs[0]?.shape);
    setShape(s);
    setSampleMode("zeros");
    setValues(Array.from({ length: s.reduce((a, b) => a * b, 1) }, () => "0").join(", "));
    try {
      setWasmInfo(await getWasmModelInfo(bytes));
    } catch (e) {
      setWasmInfoError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  // Default model: the real fraud model shipped with the site.
  useEffect(() => {
    getFraudModelBytes().then(
      (bytes) => loadModel("fraud_detector.onnx", bytes),
      (e: unknown) =>
        setLoadError(
          `Could not load ${FRAUD_MODEL_URL}: ${e instanceof Error ? e.message : String(e)}`,
        ),
    );
  }, [loadModel]);

  const needsServer = mode === "server" || mode === "both";
  const serverReady = health.status === "online" && loggedIn === true;

  const refreshServerModels = useCallback(async () => {
    setServerError(null);
    try {
      const models = await fetchModels();
      setServerModels(models);
      return models;
    } catch (e) {
      setServerError(e instanceof Error ? e.message : String(e));
      return [];
    }
  }, []);

  useEffect(() => {
    if (needsServer && serverReady) void refreshServerModels();
  }, [needsServer, serverReady, refreshServerModels]);

  useEffect(() => {
    if (!selectedModelId || !serverReady) {
      setSelectedServerModel(null);
      return;
    }
    getModel(selectedModelId).then(
      (m) => {
        setSelectedServerModel(m);
        // Server-only runs use the selected model's declared input shape.
        if (mode === "server" && m.inputs[0]) {
          const s = concreteShape(m.inputs[0].shape);
          setShape(s);
          setSampleMode("zeros");
          setValues(Array.from({ length: s.reduce((a, b) => a * b, 1) }, () => "0").join(", "));
        }
      },
      (e: unknown) => {
        setSelectedServerModel(null);
        setServerError(e instanceof Error ? e.message : String(e));
      },
    );
    // Re-run only when the selection changes, not when the mode toggles.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedModelId, serverReady]);

  const parsedValues = useMemo(
    () =>
      values
        .split(/[\s,]+/)
        .filter(Boolean)
        .map(Number),
    [values],
  );
  const expectedSize = shape.reduce((a, b) => a * b, 1);
  const nonFinite = parsedValues.some((v) => !Number.isFinite(v));
  const countOk = parsedValues.length === expectedSize;
  const valid = countOk && !nonFinite;

  // Static dims declared by the model that the chosen shape violates (engines will reject these).
  const shapeMismatch = useMemo(() => {
    const decl = declaredInput?.shape;
    if (!decl) return null;
    if (decl.length !== shape.length) {
      return `Model declares rank ${decl.length} ${fmtShape(decl)}; input has rank ${shape.length}.`;
    }
    const bad = decl.findIndex((d, i) => d > 0 && d !== shape[i]);
    return bad >= 0
      ? `Dimension ${bad} must be ${decl[bad]} (model declares ${fmtShape(decl)}).`
      : null;
  }, [declaredInput, shape]);

  const applySample = (m: SampleMode) => {
    setSampleMode(m);
    if (m === "zeros") setValues(Array.from({ length: expectedSize }, () => "0").join(", "));
    else if (m === "random")
      setValues(
        Array.from({ length: expectedSize }, () => (Math.random() * 2 - 1).toFixed(4)).join(", "),
      );
  };

  const handleModelFile = async (f: File) => {
    try {
      await loadModel(f.name, new Uint8Array(await f.arrayBuffer()));
    } catch (e) {
      setLoadError(`Failed to read ${f.name}: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    const f = e.dataTransfer.files[0];
    if (f) void handleModelFile(f);
  };

  /** POST /convert with the loaded bytes; the server stores it even if unsupported. */
  const registerOnServer = async () => {
    if (!modelBytes) return;
    setUploading(true);
    setServerError(null);
    try {
      const res = await uploadModel(new Blob([modelBytes.slice()]), modelName);
      await refreshServerModels();
      setSelectedModelId(res.onnx_model_id);
    } catch (e) {
      setServerError(e instanceof Error ? e.message : String(e));
    } finally {
      setUploading(false);
    }
  };

  const runWasm = async (): Promise<RunOutcome> => {
    try {
      if (!modelBytes) throw new Error("No model loaded.");
      const r = await runWasmInference(modelBytes, new Float32Array(parsedValues), shape);
      return {
        kind: "wasm",
        ok: true,
        output: r.output,
        outputShape: r.outputShape,
        backend: "crucible-wasm",
        computeMs: r.latencyMs,
      };
    } catch (e) {
      return { kind: "wasm", ok: false, error: e instanceof Error ? e.message : String(e) };
    }
  };

  const runServer = async (): Promise<RunOutcome> => {
    try {
      if (health.status === "offline") throw new Error(health.error);
      if (!loggedIn) throw new Error("Log in to run inference on the native server engine.");
      if (!selectedModelId) {
        throw new Error("No server model selected. Register the loaded model or pick one.");
      }
      const t0 = performance.now();
      const res = await runInference(selectedModelId, parsedValues, shape);
      const roundTripMs = performance.now() - t0;
      return {
        kind: "server",
        ok: true,
        output: res.output,
        outputShape: res.output_shape,
        backend: res.backend,
        computeMs: res.inference_time_ms,
        roundTripMs,
      };
    } catch (e) {
      return { kind: "server", ok: false, error: e instanceof Error ? e.message : String(e) };
    }
  };

  const run = async () => {
    if (!valid) return;
    setRunning(true);
    setOutcomes([]);
    // Each backend runs explicitly; a failure is reported, never replaced by the other backend.
    const results: RunOutcome[] = [];
    if (mode === "wasm" || mode === "both") results.push(await runWasm());
    if (mode === "server" || mode === "both") results.push(await runServer());
    setOutcomes(results);
    setRunning(false);
    setHistory((prev) =>
      [
        {
          id: crypto.randomUUID(),
          timestamp: Date.now(),
          model: modelName,
          mode,
          values: [...parsedValues],
          shape: [...shape],
          outcomes: results,
        },
        ...prev,
      ].slice(0, 5),
    );
  };

  const updateShape = (i: number, v: string) => {
    const parsed = Number.parseInt(v, 10);
    setShape((s) =>
      s.map((x, idx) => (idx !== i ? x : Number.isNaN(parsed) || parsed < 1 ? 1 : parsed)),
    );
  };

  const restoreFromHistory = (entry: HistoryEntry) => {
    setValues(entry.values.map(String).join(", "));
    setSampleMode("custom");
    setShape(entry.shape);
    setOutcomes(entry.outcomes);
    setShowHistory(false);
  };

  // Cross-backend agreement, only when both produced outputs.
  const okOutcomes = outcomes.filter((o) => o.ok && o.output);
  const maxAbsDiff =
    okOutcomes.length === 2 && okOutcomes[0].output!.length === okOutcomes[1].output!.length
      ? okOutcomes[0].output!.reduce(
          (m, v, i) => Math.max(m, Math.abs(v - okOutcomes[1].output![i])),
          0,
        )
      : null;

  const initDims = useMemo(
    () => new Map((meta?.initializers ?? []).map((t) => [t.name, t.dims])),
    [meta],
  );

  return (
    <CrucibleLayout>
      <section className="c-container">
        <div
          style={{
            marginBottom: 32,
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-start",
            flexWrap: "wrap",
            gap: 12,
          }}
        >
          <div style={{ maxWidth: 720 }}>
            <span className="c-badge c-badge-info">Developer Console</span>
            <h1 className="c-h2" style={{ fontSize: 44, marginTop: 14 }}>
              Inference Playground
            </h1>
            <p className="c-muted">
              Load an <span className="mono">.onnx</span> model, set the input tensor, and run it
              with the Rust WASM runtime in this tab or with the native C++ engine on the server.
              The WASM runtime is a separate small interpreter (MatMul, Add, Relu, Sigmoid,
              Softmax); CNNs and other ops need the native engine.
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

        {showHistory && history.length > 0 && (
          <div className="c-card" style={{ marginBottom: 24 }}>
            <h3 className="c-h3" style={{ marginBottom: 12 }}>
              Recent runs ({history.length})
            </h3>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {history.map((entry) => (
                <button
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
                    background: "none",
                    textAlign: "left",
                    font: "inherit",
                  }}
                >
                  <span
                    className="mono"
                    style={{ fontSize: 10, color: "var(--ink-muted)", flexShrink: 0 }}
                  >
                    {entry.mode.toUpperCase()}
                  </span>
                  <span className="mono" style={{ fontSize: 12, color: "var(--ink)" }}>
                    {entry.model}
                  </span>
                  <span className="c-muted" style={{ fontSize: 11, flex: 1 }}>
                    {fmtShape(entry.shape)}
                  </span>
                  {entry.outcomes.map((o) => (
                    <span
                      key={o.kind}
                      className="mono"
                      style={{ fontSize: 12, color: o.ok ? "var(--ok)" : "var(--risk)" }}
                    >
                      {o.kind}:{" "}
                      {o.ok && o.computeMs !== undefined ? `${fmtMs(o.computeMs)} ms` : "error"}
                    </span>
                  ))}
                </button>
              ))}
            </div>
          </div>
        )}

        {needsServer && (
          <>
            <BackendOfflineNotice health={health} onRetry={health.recheck}>
              WASM mode still works without the backend.
            </BackendOfflineNotice>
            <NativeUnavailableNotice health={health} />
          </>
        )}

        <div className="c-two-col">
          {/* LEFT — inference console */}
          <div className="c-card">
            <h3 className="c-h3">Inference Console</h3>

            <div style={{ marginTop: 12 }}>
              <label className="c-label">Backend</label>
              <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                {(
                  [
                    { value: "wasm", label: BACKEND_LABEL.wasm, icon: <Cpu size={13} /> },
                    { value: "server", label: BACKEND_LABEL.server, icon: <Server size={13} /> },
                    { value: "both", label: "Both (compare)", icon: <Layers size={13} /> },
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

            {needsServer && health.status === "online" && loggedIn === false && (
              <div className="c-card-dashed" style={{ marginTop: 12, padding: 14, fontSize: 13 }}>
                <LogIn size={14} style={{ verticalAlign: -2, marginRight: 6 }} />
                Server inference requires an account.{" "}
                <Link to="/login" style={{ color: "var(--trace)", fontWeight: 600 }}>
                  Log in
                </Link>{" "}
                or{" "}
                <Link to="/register" style={{ color: "var(--trace)", fontWeight: 600 }}>
                  register
                </Link>{" "}
                to register models and run them on the native engine.
              </div>
            )}

            {needsServer && serverReady && (
              <div style={{ marginTop: 12 }}>
                <label className="c-label">Server model</label>
                <select
                  className="c-select"
                  value={selectedModelId}
                  onChange={(e) => setSelectedModelId(e.target.value)}
                >
                  <option value="">-- Select one of your registered models --</option>
                  {serverModels.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name} · {m.operators_used.length} op types
                      {m.all_supported ? "" : " · unsupported"}
                    </option>
                  ))}
                </select>
                <button
                  className="c-btn c-btn-ghost"
                  onClick={registerOnServer}
                  disabled={!modelBytes || uploading}
                  style={{
                    marginTop: 8,
                    padding: "6px 10px",
                    display: "inline-flex",
                    gap: 6,
                    alignItems: "center",
                  }}
                >
                  <CloudUpload size={13} />
                  {uploading ? "Registering…" : `Register ${modelName} on the server`}
                </button>
                {selectedServerModel && (
                  <div className="c-muted" style={{ fontSize: 12, marginTop: 8 }}>
                    Server analysis of <span className="mono">{selectedServerModel.name}</span>:{" "}
                    {selectedServerModel.inputs.map((t) => (
                      <span key={t.name} className="mono">
                        {t.name} {fmtShape(t.shape)} {t.dtype}{" "}
                      </span>
                    ))}
                    {selectedServerModel.all_supported ? (
                      <span style={{ color: "var(--ok)" }}>· supported by the native engine</span>
                    ) : (
                      <ul style={{ color: "var(--risk)", margin: "6px 0 0 16px" }}>
                        {selectedServerModel.unsupported_reasons.map((r) => (
                          <li key={r}>{r}</li>
                        ))}
                      </ul>
                    )}
                  </div>
                )}
                {serverError && (
                  <div role="alert" style={{ color: "var(--risk)", fontSize: 12, marginTop: 6 }}>
                    {serverError}
                  </div>
                )}
              </div>
            )}

            <div
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
                  {modelBytes ? `${modelName} (${modelBytes.length.toLocaleString()} bytes)` : "—"}
                </span>
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept=".onnx"
                style={{ display: "none" }}
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) void handleModelFile(f);
                  e.target.value = "";
                }}
              />
            </div>
            {loadError && (
              <div role="alert" style={{ color: "var(--risk)", fontSize: 13, marginTop: 8 }}>
                {loadError}
              </div>
            )}

            <div style={{ marginTop: 18 }}>
              <label className="c-label">
                Input shape
                {declaredInput && (
                  <span className="c-muted" style={{ fontWeight: 400, textTransform: "none" }}>
                    {" "}
                    · <span className="mono">{declaredInput.name}</span> declared{" "}
                    {fmtShape(declaredInput.shape)}
                  </span>
                )}
              </label>
              <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                <span className="mono" style={{ color: "var(--ink-muted)" }}>
                  [
                </span>
                {shape.map((d, i) => (
                  <input
                    key={i}
                    className="c-input"
                    type="number"
                    min={1}
                    value={d}
                    onChange={(e) => updateShape(i, e.target.value)}
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
                float32 values.
              </p>
              {shapeMismatch && (
                <p style={{ fontSize: 12, marginTop: 4, color: "var(--warn)" }}>
                  {shapeMismatch} The engines will reject this shape.
                </p>
              )}
            </div>

            <div style={{ marginTop: 14 }}>
              <label className="c-label">Input values</label>
              <div style={{ display: "flex", gap: 6, marginBottom: 8 }}>
                {(["zeros", "random", "custom"] as SampleMode[]).map((s) => (
                  <button
                    key={s}
                    className={`c-preset ${sampleMode === s ? "active" : ""}`}
                    style={{
                      width: "auto",
                      padding: "5px 12px",
                      textTransform: "capitalize",
                      background: sampleMode === s ? "var(--forge-tint)" : undefined,
                      borderColor: sampleMode === s ? "var(--trace)" : undefined,
                    }}
                    onClick={() => applySample(s)}
                  >
                    {s}
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
                {nonFinite
                  ? "✗ Every value must be a finite number"
                  : countOk
                    ? `✓ ${parsedValues.length} values match shape ${fmtShape(shape)}`
                    : `✗ Got ${parsedValues.length} values — shape ${fmtShape(shape)} needs ${expectedSize}`}
              </p>
              {modelName === "fraud_detector.onnx" && (
                <p className="c-muted" style={{ fontSize: 11, marginTop: 4 }}>
                  The fraud model expects 7 z-score-normalised features (zeros = the training mean).
                  The Fraud Demo page does this normalisation from raw amounts for you.
                </p>
              )}
            </div>

            <button
              className="c-btn c-btn-primary c-btn-full"
              onClick={run}
              disabled={!valid || running || !modelBytes}
              style={{ marginTop: 18 }}
            >
              <Play size={15} /> {running ? "Running..." : "Run inference"}
            </button>
            {running && <div className="c-loading-bar" style={{ marginTop: 12 }} />}

            {outcomes.map((o) => (
              <OutcomePanel key={o.kind} outcome={o} />
            ))}
            {maxAbsDiff !== null && (
              <p className="c-muted c-fade-in" style={{ fontSize: 12, marginTop: 12 }}>
                Max |WASM − native| over {okOutcomes[0].output!.length} outputs:{" "}
                <span className="mono" style={{ color: "var(--trace)" }}>
                  {maxAbsDiff.toExponential(2)}
                </span>
              </p>
            )}
          </div>

          {/* RIGHT — graph inspector (parsed from the loaded bytes) */}
          <div className="c-card">
            <h3 className="c-h3">Model Graph Inspector</h3>
            {!meta ? (
              <p className="c-muted" style={{ marginTop: 12 }}>
                {loadError ? "No model loaded." : "Loading model…"}
              </p>
            ) : (
              <>
                <div className="c-grid-2" style={{ gap: 10, marginTop: 12 }}>
                  <MetaCell label="Model" value={modelName} mono />
                  <MetaCell label="Producer" value={meta.producer || "not recorded"} mono />
                  <MetaCell label="IR version" value={meta.irVersion ?? "—"} />
                  <MetaCell label="Opset (ai.onnx)" value={meta.opset ?? "—"} />
                </div>

                <div className="c-label" style={{ marginTop: 18 }}>
                  Inputs / outputs
                </div>
                <div className="mono" style={{ fontSize: 12, display: "grid", gap: 4 }}>
                  {meta.inputs.map((t) => (
                    <div key={`in-${t.name}`}>
                      <span style={{ color: "var(--ink-muted)" }}>in </span>
                      {t.name} {fmtShape(t.shape)}{" "}
                      {t.elemType !== null
                        ? (ELEM_TYPE_NAMES[t.elemType] ?? `type ${t.elemType}`)
                        : ""}
                    </div>
                  ))}
                  {meta.outputs.map((t) => (
                    <div key={`out-${t.name}`}>
                      <span style={{ color: "var(--ink-muted)" }}>out </span>
                      {t.name} {fmtShape(t.shape)}{" "}
                      {t.elemType !== null
                        ? (ELEM_TYPE_NAMES[t.elemType] ?? `type ${t.elemType}`)
                        : ""}
                    </div>
                  ))}
                </div>

                <div className="c-label" style={{ marginTop: 18 }}>
                  WASM runtime compatibility
                </div>
                {wasmInfoError ? (
                  <div style={{ fontSize: 12, color: "var(--risk)" }}>{wasmInfoError}</div>
                ) : !wasmInfo ? (
                  <div className="c-muted" style={{ fontSize: 12 }}>
                    Checking…
                  </div>
                ) : wasmInfo.supported ? (
                  <div style={{ fontSize: 12, color: "var(--ok)" }}>
                    Runnable in the browser ({wasmInfo.ops_used.join(", ")}).
                  </div>
                ) : (
                  <ul style={{ fontSize: 12, color: "var(--risk)", margin: "0 0 0 16px" }}>
                    {wasmInfo.unsupported_reasons.map((r) => (
                      <li key={r}>{r}</li>
                    ))}
                    <li style={{ color: "var(--ink-muted)", listStyle: "none", marginLeft: -16 }}>
                      Use the native server backend for this model.
                    </li>
                  </ul>
                )}

                <div className="c-label" style={{ marginTop: 18 }}>
                  Graph nodes ({meta.nodes.length}) · op types: {meta.opsUsed.join(", ")}
                </div>
                <div style={{ display: "grid", gap: 6, marginTop: 8 }}>
                  {meta.nodes.slice(0, 300).map((n, i) => (
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
                          <span style={{ color: "var(--ink-muted)", width: 30 }} className="mono">
                            #{i}
                          </span>
                          <span className="c-node-op">{n.opType}</span>
                        </div>
                        <span className="c-node-shape">→ {n.outputs.join(", ")}</span>
                      </button>
                      {expanded === i && (
                        <div id={`node-detail-${i}`} className="c-node-details c-fade-in">
                          {n.name && (
                            <div>
                              <span style={{ color: "var(--ink-muted)" }}>name: </span>
                              <span className="mono">{n.name}</span>
                            </div>
                          )}
                          <div>
                            <span style={{ color: "var(--ink-muted)" }}>inputs: </span>
                            <span className="mono" style={{ color: "var(--trace)" }}>
                              {n.inputs
                                .map((x) => {
                                  const dims = initDims.get(x);
                                  return dims ? `${x}${fmtShape(dims)}` : x;
                                })
                                .join(", ")}
                            </span>
                          </div>
                          <div>
                            <span style={{ color: "var(--ink-muted)" }}>outputs: </span>
                            <span className="mono" style={{ color: "var(--ok)" }}>
                              {n.outputs.join(", ")}
                            </span>
                          </div>
                          {n.attributes.length > 0 && (
                            <div>
                              <span style={{ color: "var(--ink-muted)" }}>attributes: </span>
                              <span className="mono" style={{ color: "var(--warn)" }}>
                                {n.attributes.join(", ")}
                              </span>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  ))}
                  {meta.nodes.length > 300 && (
                    <p className="c-muted" style={{ fontSize: 12 }}>
                      … {meta.nodes.length - 300} more nodes not listed.
                    </p>
                  )}
                </div>
              </>
            )}
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

function MetaCell({
  label,
  value,
  mono,
}: {
  label: string;
  value: string | number;
  mono?: boolean;
}) {
  return (
    <div className="c-metric" style={{ padding: 12 }}>
      <div className="c-metric-label">{label}</div>
      {mono ? (
        <div className="mono" style={{ fontSize: 13, marginTop: 6, color: "var(--trace)" }}>
          {value}
        </div>
      ) : (
        <div className="c-metric-value" style={{ fontSize: 20 }}>
          {value}
        </div>
      )}
    </div>
  );
}

function OutcomePanel({ outcome: o }: { outcome: RunOutcome }) {
  return (
    <div className="c-fade-in" style={{ marginTop: 22 }}>
      <div className="c-label">{BACKEND_LABEL[o.kind]}</div>
      {!o.ok ? (
        <div role="alert" style={{ color: "var(--risk)", fontSize: 13 }}>
          {o.error}
        </div>
      ) : (
        <>
          <div className="c-grid-3" style={{ gap: 10, marginTop: 8 }}>
            <div className="c-metric" style={{ padding: 12 }}>
              <div className="c-metric-label">
                {o.kind === "server" ? "Engine time (server)" : "WASM call (parse + run)"}
              </div>
              <div className="c-metric-value" style={{ fontSize: 18 }}>
                {fmtMs(o.computeMs!)} ms
              </div>
              {o.roundTripMs !== undefined && (
                <div className="c-metric-sub">{fmtMs(o.roundTripMs)} ms browser round trip</div>
              )}
            </div>
            <div className="c-metric" style={{ padding: 12 }}>
              <div className="c-metric-label">Output shape</div>
              <div className="c-metric-value" style={{ fontSize: 18 }}>
                {fmtShape(o.outputShape)}
              </div>
            </div>
            <div className="c-metric" style={{ padding: 12 }}>
              <div className="c-metric-label">Backend</div>
              <div className="c-metric-value" style={{ fontSize: 14, color: "var(--trace)" }}>
                {o.backend}
              </div>
            </div>
          </div>
          <pre className="c-code" style={{ marginTop: 12, maxHeight: 220, overflow: "auto" }}>
            {JSON.stringify(
              {
                output: o.output!.length > 64 ? [...o.output!.slice(0, 64), "…"] : o.output,
                output_shape: o.outputShape,
                backend: o.backend,
              },
              null,
              2,
            )}
          </pre>
          {o.output!.length > 64 && (
            <p className="c-muted" style={{ fontSize: 11 }}>
              Showing 64 of {o.output!.length} values.
            </p>
          )}
        </>
      )}
    </div>
  );
}
