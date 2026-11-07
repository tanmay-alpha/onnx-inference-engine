// Roadmap items reflect the repository state; "shipped" means it is in the code today.
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
    title: "20-operator native engine, checked against ONNX Runtime",
    detail:
      "Conv, pooling, BatchNormalization, Gemm and activations — enough for ResNet18 and MobileNetV2 — with a 70-case parity suite against ONNX Runtime.",
  },
  {
    status: "shipped",
    title: "Small Rust WASM runtime",
    detail:
      "A separate pure-Rust interpreter (MatMul, Add, Relu, Sigmoid, Softmax) built with wasm-bindgen; runs the fraud model in the browser.",
  },
  {
    status: "shipped",
    title: "C ABI, Python module and HTTP server",
    detail:
      "Shared library with a C interface (used by the Rust CLI), a pybind11 module, and a FastAPI server with per-user model registry.",
  },
  {
    status: "planned",
    title: "Close the performance gap with ONNX Runtime",
    detail:
      "Single-threaded, Crucible is currently several times slower than ONNX Runtime on ResNet18 / MobileNetV2. Threading and faster conv paths are next.",
  },
  {
    status: "planned",
    title: "Native ↔ WASM parity harness",
    detail: "Automatically compare the two runtimes' outputs on the models they both support.",
  },
  {
    status: "planned",
    title: "Quantized int8 MatMul path",
    detail: "Symmetric per-tensor quantization for smaller models.",
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
];
