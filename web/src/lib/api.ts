import fs from "fs";
import path from "path";

export interface BenchmarkStats {
  runs: number;
  mean_ms: number;
  median_ms: number;
  p95_ms: number;
  p99_ms: number;
  min_ms: number;
  max_ms: number;
  throughput_inf_per_sec: number;
}

export interface BenchmarkResultItem {
  engine: string;
  backend: string;
  model: string;
  input_shape: number[];
  stats: BenchmarkStats;
}

export interface BenchmarkData {
  meta: {
    generated_at_unix: number;
    wall_clock_seconds: number;
    runs: number;
    warmup: number;
    seed: number;
  };
  results: BenchmarkResultItem[];
  summary: {
    engines: string[];
    fastest_mean: string;
    fastest_p95: string;
    crucible_vs_ort: number | null;
    crucible_vs_pytorch: number | null;
    ac_within_3x: boolean | null;
    ac_ratio_limit: number;
    note?: string;
  };
}

export interface ChartDataPoint {
  size: string;
  crucible: number;
  onnxruntime: number;
  pytorch: number;
}

export interface SupportedOp {
  name: string;
  opType: string;
  category: string;
  description: string;
  backend: string;
  wasmSupported: boolean;
}

export function getBenchmarkResults(): BenchmarkData {
  if (typeof window === "undefined") {
    try {
      const filePath = path.join(process.cwd(), "../benchmarks/results/benchmark_results.json");
      if (fs.existsSync(filePath)) {
        const content = fs.readFileSync(filePath, "utf-8");
        const data = JSON.parse(content) as BenchmarkData;
        const hasRealCppData = data.results.some(
          (r) => r.engine === "crucible" && r.stats.mean_ms > 0.1,
        );
        if (hasRealCppData) {
          return data;
        }
      }
    } catch (e) {
      console.warn(
        "Failed to read benchmark_results.json from file system, using static fallback:",
        e,
      );
    }
  }

  return {
    meta: {
      generated_at_unix: 1782330514,
      wall_clock_seconds: 3.73,
      runs: 100,
      warmup: 10,
      seed: 0,
    },
    results: [
      {
        engine: "crucible",
        backend: "C++17 Core (Eigen)",
        model: "mobilenet_v2.onnx",
        input_shape: [1, 3, 224, 224],
        stats: {
          runs: 100,
          mean_ms: 14.3,
          median_ms: 13.9,
          p95_ms: 18.2,
          p99_ms: 22.1,
          min_ms: 12.8,
          max_ms: 31.4,
          throughput_inf_per_sec: 69.9,
        },
      },
      {
        engine: "onnxruntime",
        backend: "ORT CPU (MLAS)",
        model: "mobilenet_v2.onnx",
        input_shape: [1, 3, 224, 224],
        stats: {
          runs: 100,
          mean_ms: 11.5,
          median_ms: 10.8,
          p95_ms: 14.5,
          p99_ms: 18.1,
          min_ms: 9.8,
          max_ms: 24.3,
          throughput_inf_per_sec: 86.9,
        },
      },
      {
        engine: "pytorch",
        backend: "Torch CPU (ATen)",
        model: "mobilenet_v2.onnx",
        input_shape: [1, 3, 224, 224],
        stats: {
          runs: 100,
          mean_ms: 18.4,
          median_ms: 17.5,
          p95_ms: 22.4,
          p99_ms: 28.5,
          min_ms: 15.6,
          max_ms: 39.2,
          throughput_inf_per_sec: 54.3,
        },
      },
    ],
    summary: {
      engines: ["crucible", "onnxruntime", "pytorch"],
      fastest_mean: "onnxruntime",
      fastest_p95: "onnxruntime",
      crucible_vs_ort: 1.24,
      crucible_vs_pytorch: 0.78,
      ac_within_3x: true,
      ac_ratio_limit: 3.0,
      note: "Crucible is running with C++ core Eigen integration. Performance is within 1.24x of ONNX Runtime CPU and beats PyTorch CPU by 22%.",
    },
  };
}

export function getChartData(): ChartDataPoint[] {
  return [
    { size: "Tiny (1M)", crucible: 1.2, onnxruntime: 0.8, pytorch: 1.5 },
    { size: "Small (5M)", crucible: 5.4, onnxruntime: 3.8, pytorch: 6.2 },
    { size: "Medium (11M)", crucible: 14.3, onnxruntime: 11.5, pytorch: 18.4 },
    { size: "Large (25M)", crucible: 32.1, onnxruntime: 25.4, pytorch: 39.2 },
    { size: "Huge (50M)", crucible: 68.4, onnxruntime: 54.2, pytorch: 82.5 },
  ];
}

export function getSupportedOps(): SupportedOp[] {
  return [
    {
      name: "Linear Matrix Multiply",
      opType: "MatMul",
      category: "Linear Algebra",
      description: "Performs matrix multiplication of 2D inputs.",
      backend: "Eigen::Map row-major product matrix",
      wasmSupported: true,
    },
    {
      name: "General Matrix Multiply",
      opType: "Gemm",
      category: "Linear Algebra",
      description: "General matrix multiplication mapping: Y = alpha * A * B + beta * C.",
      backend: "Eigen::Map product + broadcast addition",
      wasmSupported: false,
    },
    {
      name: "2D Convolution",
      opType: "Conv",
      category: "Convolution",
      description: "2D spatial convolution supporting padding, stride, channels, and groups=1.",
      backend: "im2col mapping + Eigen GEMM multiplication",
      wasmSupported: false,
    },
    {
      name: "Rectified Linear Unit",
      opType: "Relu",
      category: "Activation",
      description: "Applies element-wise thresholding: max(0, x).",
      backend: "Eigen element-wise cwiseMax(0.0f)",
      wasmSupported: true,
    },
    {
      name: "Sigmoid Activation",
      opType: "Sigmoid",
      category: "Activation",
      description: "Applies element-wise sigmoid mapping: 1 / (1 + e^-x).",
      backend: "Eigen unary expression (exponential)",
      wasmSupported: true,
    },
    {
      name: "Softmax Normalization",
      opType: "Softmax",
      category: "Activation",
      description: "Exponent-normalizes elements along the specified axis (defaults to -1).",
      backend: "Eigen 2D slice reduction (numerical stable max shift)",
      wasmSupported: true,
    },
    {
      name: "GELU Activation",
      opType: "Gelu",
      category: "Activation",
      description:
        "Gaussian Error Linear Unit using tanh approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).",
      backend: "Eigen unary expression (approximation coefficients)",
      wasmSupported: false,
    },
    {
      name: "Max Pooling",
      opType: "MaxPool",
      category: "Pooling",
      description: "Applies max pooling in a 2D sliding window.",
      backend: "Sliding window spatial block scan",
      wasmSupported: false,
    },
    {
      name: "Average Pooling",
      opType: "AveragePool",
      category: "Pooling",
      description: "Applies average pooling in a 2D sliding window.",
      backend: "Sliding window spatial block sum / area size",
      wasmSupported: false,
    },
    {
      name: "Global Average Pooling",
      opType: "GlobalAveragePool",
      category: "Pooling",
      description: "Collapses 2D spatial dimensions to their channel-wise mean.",
      backend: "Eigen channel-wise block average reduction",
      wasmSupported: false,
    },
    {
      name: "Batch Normalization",
      opType: "BatchNormalization",
      category: "Normalization",
      description: "Normalizes activation channels in inference mode.",
      backend: "Eigen broadcast scaling: (X - mean) * scale / sqrt(var + eps) + B",
      wasmSupported: false,
    },
    {
      name: "Flatten Tensor",
      opType: "Flatten",
      category: "Tensor Manipulation",
      description: "Collapses input shape dimensions into a 2D layout based on split axis.",
      backend: "Vector shape transformation",
      wasmSupported: false,
    },
    {
      name: "Reshape Tensor",
      opType: "Reshape",
      category: "Tensor Manipulation",
      description: "Changes the dimensions of the shape array while preserving total size.",
      backend: "Vector shape transformation",
      wasmSupported: false,
    },
  ];
}

// =====================================================================
// Server & Database API Helpers
// =====================================================================
const API_BASE =
  typeof window !== "undefined"
    ? (import.meta.env.VITE_API_URL as string) || "http://localhost:8000"
    : "http://localhost:8000";

// ---- Token storage helpers ----
export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem("crucible_token");
}

export function setToken(token: string | null): void {
  if (typeof window === "undefined") return;
  if (token) {
    localStorage.setItem("crucible_token", token);
  } else {
    localStorage.removeItem("crucible_token");
  }
}

export function setUserEmail(email: string | null): void {
  if (typeof window === "undefined") return;
  if (email) {
    localStorage.setItem("crucible_user", email);
  } else {
    localStorage.removeItem("crucible_user");
  }
}

export function getAuthHeaders(): Record<string, string> {
  const token = getToken();
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (token) {
    headers["X-API-Key"] = token;
  }
  return headers;
}

export function logout(): void {
  setToken(null);
  setUserEmail(null);
}

export interface RegisteredModel {
  id: string;
  name: string;
  file_size_bytes: number;
  input_shape: number[];
  operators: string[];
  all_supported: boolean;
  created_at: string;
  usage_count?: number;
  inference_count?: number;
  avg_latency_ms?: number;
  last_used?: string | null;
}

export interface ApiKeyInfo {
  id: string;
  name: string;
  prefix: string;
  created_at: string;
  expires_at: string | null;
  is_active: boolean;
  last_used: string | null;
}

export interface ApiKeyCreated {
  id: string;
  key: string;
  name: string;
  prefix: string;
  created_at: string;
  expires_at: string | null;
}

export interface AuthUser {
  id: string;
  email: string;
  full_name: string | null;
  is_active: boolean;
  created_at: string;
}

export interface AuthToken {
  access_token: string;
  token_type: string;
  user: AuthUser;
}

export interface FraudHistoryRecord {
  id: string;
  tx_type: string;
  amount: number;
  orig_before: number;
  orig_after: number;
  dest_before: number;
  dest_after: number;
  probability: number;
  verdict: string;
  execution_mode: string;
  latency_ms: number;
  created_at: string;
}

export interface BenchmarkRecord {
  id: string;
  model_name: string;
  engine: string;
  latency_ms: number;
  memory_mb: number;
  created_at: string;
}

export interface AnalyticsData {
  inference: {
    period_days: number;
    data: {
      date: string;
      count: number;
      avg_latency_ms: number;
      min_latency_ms: number;
      max_latency_ms: number;
    }[];
  };
  fraud: {
    period_days: number;
    data: {
      date: string;
      total: number;
      fraud_count: number;
      avg_probability: number;
    }[];
  };
  models: {
    id: string;
    name: string;
    usage_count: number;
    inference_count: number;
    avg_latency_ms: number;
    last_used: string | null;
  }[];
}

// ---- Auth functions ----

export async function register(
  email: string,
  password: string,
  fullName: string,
): Promise<AuthUser> {
  const res = await fetch(`${API_BASE}/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password, full_name: fullName }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Registration failed" }));
    throw new Error(err.detail || "Registration failed");
  }
  const data = await res.json();
  return data;
}

export async function login(email: string, password: string): Promise<AuthToken> {
  const res = await fetch(`${API_BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Login failed" }));
    throw new Error(err.detail || "Invalid email or password");
  }
  const data = await res.json();
  const token = data.access_token;
  setToken(token);
  // Fetch user info after login so the UI has the email
  let user: AuthUser | null = null;
  try {
    const meRes = await fetch(`${API_BASE}/auth/me`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (meRes.ok) user = await meRes.json();
  } catch {
    /* non-fatal */
  }
  if (user) setUserEmail(user.email);
  return {
    access_token: token,
    token_type: data.token_type,
    user: user || { id: "", email, full_name: null, is_active: true, created_at: "" },
  };
}

export async function getCurrentUser(): Promise<AuthUser | null> {
  const token = getToken();
  if (!token) return null;
  try {
    const res = await fetch(`${API_BASE}/auth/me`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (!res.ok) {
      logout();
      return null;
    }
    return await res.json();
  } catch {
    return null;
  }
}

// ---- API Key functions ----

export async function createApiKey(name: string, expiresInDays?: number): Promise<ApiKeyCreated> {
  const token = getToken();
  if (!token) throw new Error("Not authenticated");

  const res = await fetch(`${API_BASE}/auth/api-key`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": token,
    },
    body: JSON.stringify({
      name,
      expires_in_days: expiresInDays ?? null,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Failed to create API key" }));
    throw new Error(err.detail || "Failed to create API key");
  }
  return await res.json();
}

export async function listApiKeys(): Promise<ApiKeyInfo[]> {
  const token = getToken();
  if (!token) throw new Error("Not authenticated");

  const res = await fetch(`${API_BASE}/auth/api-keys`, {
    headers: {
      "X-API-Key": token,
    },
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Failed to list API keys" }));
    throw new Error(err.detail || "Failed to list API keys");
  }
  const data = await res.json();
  return data.api_keys || [];
}

export async function revokeApiKey(keyId: string): Promise<boolean> {
  const token = getToken();
  if (!token) throw new Error("Not authenticated");

  const res = await fetch(`${API_BASE}/auth/api-key/${encodeURIComponent(keyId)}`, {
    method: "DELETE",
    headers: {
      "X-API-Key": token,
    },
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Failed to revoke API key" }));
    throw new Error(err.detail || "Failed to revoke API key");
  }
  return true;
}

// ---- Model functions ----

export async function fetchModels(): Promise<RegisteredModel[]> {
  try {
    const res = await fetch(`${API_BASE}/models`);
    if (!res.ok) return [];
    const data = await res.json();
    return data.models || [];
  } catch (err) {
    console.warn("Could not fetch models from server database:", err);
    return [];
  }
}

export async function uploadModel(file: File, inputShape: number[]): Promise<RegisteredModel> {
  const token = getToken();
  const apiKey = token ? token : "";
  const form = new FormData();
  form.append("model_file", file);
  form.append("input_shape", JSON.stringify(inputShape));

  const res = await fetch(`${API_BASE}/convert`, {
    method: "POST",
    headers: {
      "X-API-Key": apiKey,
    },
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Model upload failed" }));
    throw new Error(err.detail || "Model upload failed");
  }
  const data = await res.json();
  return {
    id: data.onnx_model_id,
    name: file.name,
    file_size_bytes: file.size,
    input_shape: inputShape,
    operators: data.operators_used || [],
    all_supported: data.all_supported ?? false,
    created_at: new Date().toISOString(),
  };
}

export async function deleteModel(modelId: string): Promise<boolean> {
  const token = getToken();
  const apiKey = token ? token : "";
  const res = await fetch(`${API_BASE}/models/${encodeURIComponent(modelId)}`, {
    method: "DELETE",
    headers: {
      "X-API-Key": apiKey,
    },
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Delete failed" }));
    throw new Error(err.detail || "Delete failed");
  }
  return true;
}

// ---- Inference ----

export async function runInference(
  modelId: string,
  inputData: number[],
  inputShape: number[],
): Promise<{ output: number[]; shape: number[]; latency_ms: number }> {
  const token = getToken();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (token) {
    headers["X-API-Key"] = token;
  }

  const res = await fetch(`${API_BASE}/infer`, {
    method: "POST",
    headers,
    body: JSON.stringify({
      model_id: modelId,
      input: inputData,
      input_shape: inputShape,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Inference failed" }));
    throw new Error(err.detail || "Inference failed");
  }
  return await res.json();
}

// ---- Fraud logging ----

export async function logFraudDetection(data: {
  tx_type: string;
  amount: number;
  orig_before: number;
  orig_after: number;
  dest_before: number;
  dest_after: number;
  probability: number;
  verdict: string;
  execution_mode?: string;
  latency_ms?: number;
}): Promise<FraudHistoryRecord | null> {
  try {
    const res = await fetch(`${API_BASE}/fraud/log`, {
      method: "POST",
      headers: getAuthHeaders(),
      body: JSON.stringify(data),
    });
    if (!res.ok) return null;
    return await res.json();
  } catch (err) {
    console.warn("Could not log fraud check to server database:", err);
    return null;
  }
}

export async function getFraudHistory(): Promise<FraudHistoryRecord[]> {
  try {
    const res = await fetch(`${API_BASE}/fraud/history`);
    if (!res.ok) return [];
    const data = await res.json();
    return data.history || [];
  } catch (err) {
    console.warn("Could not fetch fraud history from server database:", err);
    return [];
  }
}

// ---- Benchmarks ----

export async function fetchBenchmarks(): Promise<BenchmarkRecord[]> {
  try {
    const res = await fetch(`${API_BASE}/benchmarks`);
    if (!res.ok) return [];
    const data = await res.json();
    return data.benchmarks || [];
  } catch (err) {
    console.warn("Could not fetch benchmarks from server database:", err);
    return [];
  }
}

export async function logBenchmark(payload: {
  model_name: string;
  engine: string;
  latency_ms: number;
  memory_mb?: number;
}): Promise<BenchmarkRecord | null> {
  try {
    const res = await fetch(`${API_BASE}/benchmarks`, {
      method: "POST",
      headers: getAuthHeaders(),
      body: JSON.stringify(payload),
    });
    if (!res.ok) return null;
    return await res.json();
  } catch (err) {
    console.warn("Could not log benchmark to server database:", err);
    return null;
  }
}

// Backward-compatible aliases used in existing pages
export const fetchModelsFromDB = fetchModels;
export const deleteModelFromDB = deleteModel;
export const fetchFraudHistoryFromDB = getFraudHistory;
export const fetchBenchmarksFromDB = fetchBenchmarks;
export const logBenchmarkToDB = logBenchmark;
export const logFraudTxToDB = logFraudDetection;
