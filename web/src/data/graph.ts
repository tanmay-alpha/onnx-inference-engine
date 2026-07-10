// The exact graph the hero animates through — matches Playground's fraud_mlp_v3.
// Kept as typed data so it can be swapped for a decoded ONNX graph later.
export interface GraphOp {
  id: string;
  op: string;
  inShape: string;
  outShape: string;
  cost: number; // per-op cost in µs; sums to the sample latency
}

export const HERO_GRAPH: GraphOp[] = [
  { id: "n0", op: "MatMul", inShape: "[1,7]·[7,16]", outShape: "[1,16]", cost: 210 },
  { id: "n1", op: "Add", inShape: "[1,16]·[16]", outShape: "[1,16]", cost: 40 },
  { id: "n2", op: "Relu", inShape: "[1,16]", outShape: "[1,16]", cost: 30 },
  { id: "n3", op: "MatMul", inShape: "[1,16]·[16,8]", outShape: "[1,8]", cost: 320 },
  { id: "n4", op: "Add", inShape: "[1,8]·[8]", outShape: "[1,8]", cost: 30 },
  { id: "n5", op: "Relu", inShape: "[1,8]", outShape: "[1,8]", cost: 25 },
  { id: "n6", op: "MatMul", inShape: "[1,8]·[8,1]", outShape: "[1,1]", cost: 470 },
  { id: "n7", op: "Sigmoid", inShape: "[1,1]", outShape: "[1,1]", cost: 55 },
];
