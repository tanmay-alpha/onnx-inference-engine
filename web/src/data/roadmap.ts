// Roadmap items are typed so real data can swap in later.
export type RoadmapStatus = "shipped" | "in-progress" | "planned";

export interface RoadmapItem {
  title: string;
  status: RoadmapStatus;
  detail: string;
}

export const ROADMAP: RoadmapItem[] = [
  {
    status: "shipped",
    title: "C++17 core with custom tensor class",
    detail: "Row-major flat memory, shape/stride bookkeeping, Eigen-backed matmul and conv2d.",
  },
  {
    status: "shipped",
    title: "Hand-written ONNX protobuf decoder",
    detail:
      "Reads .onnx binary wire format without protoc. Varint, length-delimited fields, embedded tensors.",
  },
  {
    status: "shipped",
    title: "Kahn's algorithm graph executor",
    detail: "Topological sort over the DAG, ready-set scheduling, deterministic execution order.",
  },
  {
    status: "shipped",
    title: "Pure-Rust WASM reimplementation",
    detail:
      "Second implementation of the kernels in Rust, compiled to wasm-simd128, no Eigen dependency.",
  },
  {
    status: "in-progress",
    title: "Kernel parity harness",
    detail:
      "Fixed-seed golden tests comparing every C++ kernel output against the Rust kernel bit-for-bit.",
  },
  {
    status: "in-progress",
    title: "Quantized int8 MatMul path",
    detail: "Symmetric per-tensor quantization to shrink the fraud model from 220 B to ~64 B.",
  },
  {
    status: "planned",
    title: "Biometric matching (face embeddings)",
    detail:
      "On-device MobileFaceNet inference; cosine similarity in the tab, no biometric data uploaded.",
  },
  {
    status: "planned",
    title: "Document OCR (CRNN)",
    detail:
      "Client-side ID/receipt OCR — pixels never leave the device, only extracted fields cross the wire.",
  },
  {
    status: "planned",
    title: "WebGPU backend",
    detail:
      "Optional GPU kernels for the larger conv-heavy models; keep WASM as the always-available fallback.",
  },
  {
    status: "planned",
    title: "Streaming model loader",
    detail: "Range-request the .onnx from CDN and start executing before the full file has landed.",
  },
  {
    status: "planned",
    title: "Native binary distribution",
    detail:
      "Ship the C++ engine as a static library for edge / embedded targets outside the browser.",
  },
];
