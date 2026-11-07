// =====================================================================
// Crucible server API client.
//
// Every protected request authenticates with `Authorization: Bearer <JWT>`
// (the token returned by POST /auth/login). Network failures surface as
// BackendOfflineError so pages can show an explicit "backend offline" state
// instead of breaking. Nothing here fabricates data: when the server cannot
// answer, callers get an error, not a fallback.
// =====================================================================

export const API_BASE: string = (
  (import.meta.env.VITE_API_URL as string | undefined) || "http://localhost:8000"
).replace(/\/+$/, "");

export class BackendOfflineError extends Error {
  constructor() {
    super(`Backend offline: could not reach the Crucible API at ${API_BASE}.`);
    this.name = "BackendOfflineError";
  }
}

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

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

/** Auth headers for protected endpoints (JWT as a Bearer token). */
export function getAuthHeaders(json = true): Record<string, string> {
  const headers: Record<string, string> = {};
  if (json) headers["Content-Type"] = "application/json";
  const token = getToken();
  if (token) headers.Authorization = `Bearer ${token}`;
  return headers;
}

export function logout(): void {
  setToken(null);
  setUserEmail(null);
}

function errorDetail(body: unknown, fallback: string): string {
  if (body && typeof body === "object" && "detail" in body) {
    const d = (body as { detail: unknown }).detail;
    if (typeof d === "string") return d;
    if (Array.isArray(d)) {
      return d
        .map((e) =>
          e && typeof e === "object" && "msg" in e
            ? String((e as { msg: unknown }).msg)
            : String(e),
        )
        .join("; ");
    }
  }
  return fallback;
}

/** fetch() against the API. Throws BackendOfflineError / ApiError; returns parsed JSON. */
async function apiFetch<T>(path: string, init: RequestInit, fallbackError: string): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, init);
  } catch {
    throw new BackendOfflineError();
  }
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    if (res.status === 401) {
      throw new ApiError(errorDetail(body, "Not authenticated — please log in."), 401);
    }
    throw new ApiError(`${errorDetail(body, fallbackError)} (HTTP ${res.status})`, res.status);
  }
  return (await res.json()) as T;
}

// ---- Types (mirror the server contract) ----

export interface TensorInfo {
  name: string;
  /** -1 marks a symbolic dimension; null when the model declares no shape. */
  shape: number[] | null;
  dtype: string;
}

export interface RegisteredModel {
  id: string;
  name: string;
  size_bytes: number;
  created_at: string;
  opset: number | null;
  operators_used: string[];
  all_supported: boolean;
  unsupported_reasons: string[];
  inputs: TensorInfo[];
  outputs: TensorInfo[];
}

export interface ConvertResult {
  onnx_model_id: string;
  name: string;
  size_bytes: number;
  opset: number | null;
  operators_used: string[];
  all_supported: boolean;
  unsupported_ops: string[];
  unsupported_reasons: string[];
  inputs: TensorInfo[];
  outputs: TensorInfo[];
}

export interface InferResult {
  model_id: string;
  output: number[];
  output_shape: number[];
  /** Server-side engine time only (perf_counter around crucible_py.run). */
  inference_time_ms: number;
  backend: string;
  engine: string;
}

export interface HealthInfo {
  status: string;
  engine: string;
  native_available: boolean;
  version: string;
}

export interface ApiKeyInfo {
  id: string;
  name: string;
  key_prefix: string;
  is_active: boolean;
  rate_limit: number;
  last_used: string | null;
  created_at: string;
  expires_at: string | null;
}

export interface ApiKeyCreated extends ApiKeyInfo {
  /** The full key; returned only once, at creation. */
  full_key: string;
}

export interface InferenceLog {
  id: string;
  model_id: string;
  latency_ms: number;
  backend: string;
  input_shape: number[];
  output_shape: number[];
  status: string;
  created_at: string;
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

export interface InferenceAnalytics {
  period_days: number;
  data: {
    date: string;
    count: number;
    avg_latency_ms: number;
    min_latency_ms: number;
    max_latency_ms: number;
  }[];
}

export interface FraudAnalytics {
  period_days: number;
  data: {
    date: string;
    total: number;
    fraud_count: number;
    avg_probability: number;
  }[];
}

export interface ModelUsage {
  id: string;
  name: string;
  usage_count: number;
  inference_count: number;
  avg_latency_ms: number;
  last_used: string | null;
}

// ---- Health ----

/** GET /health. Throws BackendOfflineError when the API is unreachable. */
export async function getHealth(): Promise<HealthInfo> {
  return apiFetch<HealthInfo>("/health", {}, "Health check failed");
}

// ---- Auth ----

export async function register(
  email: string,
  password: string,
  fullName: string,
): Promise<AuthUser> {
  return apiFetch<AuthUser>(
    "/auth/register",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password, full_name: fullName }),
    },
    "Registration failed",
  );
}

export async function login(email: string, password: string): Promise<AuthToken> {
  const data = await apiFetch<{ access_token: string; token_type: string }>(
    "/auth/login",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    },
    "Invalid email or password",
  );
  setToken(data.access_token);
  let user: AuthUser | null = null;
  try {
    user = await apiFetch<AuthUser>("/auth/me", { headers: getAuthHeaders(false) }, "");
  } catch {
    /* non-fatal: the UI falls back to the typed email */
  }
  setUserEmail(user?.email ?? email);
  return {
    access_token: data.access_token,
    token_type: data.token_type,
    user: user || { id: "", email, full_name: null, is_active: true, created_at: "" },
  };
}

export async function getCurrentUser(): Promise<AuthUser | null> {
  if (!getToken()) return null;
  try {
    return await apiFetch<AuthUser>("/auth/me", { headers: getAuthHeaders(false) }, "");
  } catch (e) {
    if (e instanceof ApiError && e.status === 401) logout();
    return null;
  }
}

// ---- API keys ----

export async function createApiKey(name: string, expiresInDays?: number): Promise<ApiKeyCreated> {
  return apiFetch<ApiKeyCreated>(
    "/auth/api-key",
    {
      method: "POST",
      headers: getAuthHeaders(),
      body: JSON.stringify({ name, expires_in_days: expiresInDays ?? null }),
    },
    "Failed to create API key",
  );
}

export async function listApiKeys(): Promise<ApiKeyInfo[]> {
  return apiFetch<ApiKeyInfo[]>(
    "/auth/api-keys",
    { headers: getAuthHeaders(false) },
    "Failed to list API keys",
  );
}

export async function revokeApiKey(keyId: string): Promise<void> {
  await apiFetch<unknown>(
    `/auth/api-key/${encodeURIComponent(keyId)}`,
    { method: "DELETE", headers: getAuthHeaders(false) },
    "Failed to revoke API key",
  );
}

// ---- Models ----

export async function fetchModels(): Promise<RegisteredModel[]> {
  const data = await apiFetch<{ models: RegisteredModel[] }>(
    "/models",
    { headers: getAuthHeaders(false) },
    "Failed to list models",
  );
  return data.models;
}

export async function getModel(modelId: string): Promise<RegisteredModel> {
  return apiFetch<RegisteredModel>(
    `/models/${encodeURIComponent(modelId)}`,
    { headers: getAuthHeaders(false) },
    "Failed to load model",
  );
}

/** POST /convert — stores the model (even if unsupported) and returns its static analysis. */
export async function uploadModel(file: File | Blob, fileName: string): Promise<ConvertResult> {
  const form = new FormData();
  form.append("model_file", file, fileName);
  return apiFetch<ConvertResult>(
    "/convert",
    { method: "POST", headers: getAuthHeaders(false), body: form },
    "Model upload failed",
  );
}

export async function deleteModel(modelId: string): Promise<void> {
  await apiFetch<unknown>(
    `/models/${encodeURIComponent(modelId)}`,
    { method: "DELETE", headers: getAuthHeaders(false) },
    "Delete failed",
  );
}

// ---- Inference ----

export async function runInference(
  modelId: string,
  inputData: number[],
  inputShape: number[],
): Promise<InferResult> {
  return apiFetch<InferResult>(
    "/infer",
    {
      method: "POST",
      headers: getAuthHeaders(),
      body: JSON.stringify({ model_id: modelId, input: inputData, input_shape: inputShape }),
    },
    "Inference failed",
  );
}

/** GET /inference/logs — the caller's own inference history (engine latency per call). */
export async function getInferenceLogs(limit = 200): Promise<InferenceLog[]> {
  const data = await apiFetch<{ logs: InferenceLog[] }>(
    `/inference/logs?limit=${limit}`,
    { headers: getAuthHeaders(false) },
    "Failed to load inference logs",
  );
  return data.logs;
}

// ---- Fraud logging (requires login) ----

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
}): Promise<FraudHistoryRecord> {
  return apiFetch<FraudHistoryRecord>(
    "/fraud/log",
    { method: "POST", headers: getAuthHeaders(), body: JSON.stringify(data) },
    "Failed to log fraud check",
  );
}

export async function getFraudHistory(): Promise<FraudHistoryRecord[]> {
  const data = await apiFetch<{ history: FraudHistoryRecord[] }>(
    "/fraud/history",
    { headers: getAuthHeaders(false) },
    "Failed to load fraud history",
  );
  return data.history;
}

// ---- Analytics (dashboard) ----

export async function getInferenceAnalytics(days: number): Promise<InferenceAnalytics> {
  return apiFetch<InferenceAnalytics>(
    `/analytics/inference?days=${days}`,
    { headers: getAuthHeaders(false) },
    "Failed to load inference analytics",
  );
}

export async function getFraudAnalytics(days: number): Promise<FraudAnalytics> {
  return apiFetch<FraudAnalytics>(
    `/analytics/fraud?days=${days}`,
    { headers: getAuthHeaders(false) },
    "Failed to load fraud analytics",
  );
}

export async function getModelUsage(): Promise<ModelUsage[]> {
  const data = await apiFetch<{ models: ModelUsage[] }>(
    "/analytics/models",
    { headers: getAuthHeaders(false) },
    "Failed to load model usage",
  );
  return data.models;
}

// ---- Benchmarks (static file produced by `python benchmarks/run_all.py`) ----

export type BenchmarkRuntime =
  | "crucible-native"
  | "crucible-native-avx2"
  | "onnxruntime-cpu"
  | "onnxruntime-cpu-1t"
  | "pytorch-cpu"
  | "crucible-wasm";

export interface BenchmarkResult {
  model: string;
  input_shape: number[];
  runtime: BenchmarkRuntime | string;
  threads: number;
  status: "ok" | "unsupported";
  mean_ms: number | null;
  median_ms: number | null;
  p95_ms: number | null;
  min_ms: number | null;
  std_ms: number | null;
  throughput_ips: number | null;
  load_ms: number | null;
  max_abs_diff_vs_ort: number | null;
  top1_match: boolean | null;
  note: string | null;
}

export interface BenchmarkFile {
  schema: number;
  generated_at: string;
  git_commit: string;
  environment: {
    cpu: string;
    cores: number;
    os: string;
    python: string;
    compiler: string;
    onnxruntime: string;
    torch: string | null;
    crucible_build: string;
  };
  methodology: { warmup: number; iterations: number; timer: string; notes: string[] };
  results: BenchmarkResult[];
  artifacts: { name: string; bytes: number }[];
}

/**
 * Fetch the measured benchmark file shipped with the site. Returns null when
 * the file has not been generated (HTTP 404 or a non-JSON fallback page).
 */
export async function getBenchmarkFile(): Promise<BenchmarkFile | null> {
  const res = await fetch("/benchmarks/results.json", { cache: "no-store" });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`Failed to load /benchmarks/results.json (HTTP ${res.status})`);
  const type = res.headers.get("content-type") || "";
  if (!type.includes("json")) return null;
  const data = (await res.json()) as BenchmarkFile;
  if (data.schema !== 1 || !Array.isArray(data.results)) {
    throw new Error(`Unsupported benchmark file schema: ${String(data.schema)}`);
  }
  return data;
}
