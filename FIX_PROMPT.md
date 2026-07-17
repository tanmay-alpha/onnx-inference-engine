# Crucible — Full Fix Prompt for Claude Code

You are working on the **Crucible** project at `C:\Users\TANMAY\OneDrive\Desktop\Crucible`. This is a multi-component ML inference project with a Rust CLI, C++ engine (scaffolded), Python FastAPI server, WASM runtime, and web frontend. You have full permissions to modify any file.

Your job is to fix **every bug and issue** found during the audit. Work through them in order. After each fix, verify the change compiles/tests cleanly before moving on.

---

## Section 1: Models (do first — these are root causes for downstream components)

### 1.1 Fix the fraud model data leak
**File:** `models/fraud/train_fraud_model.py`, line 31

**Problem:** `neworg_f = np.zeros(n_fraud, dtype=np.float32)` hardcodes all fraud samples to have `newbalanceOrig = 0.0`, while legitimate samples have realistic values (mean ~74,709). The logistic regression uses this as a perfect discriminator, producing AUC=1.0 which is meaningless.

**Fix:** Replace the zero array with a realistic distribution. Generate `newbalanceOrig` for fraud samples using a similar random distribution as legitimate samples (e.g., `np.random.uniform(0, max_balance, n_fraud)`), possibly with a different mean/range to simulate the pattern that actual fraud might show (e.g., accounts drained to near-zero). The key is that fraud samples should NOT all have exactly 0.0 — they should have a distribution that the model must actually learn to distinguish.

Also update the `model_config.json` AUC value after retraining to reflect the real (lower) score.

### 1.2 Fix the ONNX opset API call
**File:** `models/fraud/train_fraud_model.py`, line 77

**Problem:** `helper.make_opsetid("", 13)` — the function `make_opsetid` does not exist in `onnx.helper`. The correct name is `make_operatorsetid`.

**Fix:** Change to `helper.make_operatorsetid("", 13)`.

### 1.3 Fix the output shape inconsistency
**File:** `models/fraud/train_fraud_model.py`, line 74

**Problem:** Output shape is `[None, 1]` but the MatMul+Add+Sigmoid chain naturally produces shape `[batch]` (scalar per sample). The graph is self-inconsistent.

**Fix:** Change the output to `[None]` to match what the Sigmoid node actually produces: `[helper.make_tensor_value_info("prob", TensorProto.FLOAT, [None])]`.

### 1.4 Fix fixture generation path
**File:** `models/generate_fixtures.py`, lines 24-25, 123

**Problems:**
- `REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` assumes `models/` is exactly two levels below repo root — fragile with symlinks.
- `os.makedirs(FIXTURES_DIR, exist_ok=True)` is called inside `write_model()` on every call (5 times).

**Fix:**
- Add validation at the top of `main()`: assert that `engine/tests/fixtures/` exists or can be created under the computed `REPO_ROOT`. If `REPO_ROOT/engine` doesn't exist, raise a clear error explaining the expected layout.
- Move `os.makedirs` to the top of `main()` before the loop.

### 1.5 Add byte-order and shape validation to fixtures
**File:** `models/generate_fixtures.py`, lines 28-67

**Fix:** Before writing each fixture, validate that (a) all tensor shapes have non-empty, non-negative dimensions, (b) the numpy array byte order is native (add `.newbyteorder('=')` or `.byteswap().newbyteorder()` if needed), and (c) the opset version 17 is compatible with each node type used.

---

## Section 2: Rust CLI

### 2.1 Add null pointer guard in formatter
**File:** `cli/src/formatter.rs`, lines 24-31 and 41-46

**Problem:** `*info.input_names.offset(i)` dereferences the outer `*const *const c_char` pointer without checking if `info.input_names` itself is null. This is inside an `unsafe` block and is undefined behavior if the C side returns a null pointer with `num_inputs > 0`.

**Fix:** Before the loop, add:
```rust
if info.input_names.is_null() {
    // handle gracefully — print empty names or return an error
}
```
Same for `info.output_names`.

### 2.2 Validate i32 fields from C before casting to usize
**File:** `cli/src/runner.rs`, lines 242 and 287

**Problem:** `info.num_outputs as usize` and `od.rank as usize` — if the C library returns negative values (corrupted model or bug), the cast produces `usize::MAX`, causing OOM or panic in the subsequent `vec!` allocation.

**Fix:** After reading from C, validate:
```rust
let n_out = info.num_outputs;
if n_out < 0 {
    return Err(Error::Internal(format!("C library returned negative num_outputs: {}", n_out)));
}
let n_out = n_out as usize;
```
Same pattern for `od.rank`.

### 2.3 Filter NaN values before computing benchmark statistics
**File:** `cli/src/runner.rs`, lines 281-283

**Problem:** `partial_cmp` returns `None` for NaN, converted to `Ordering::Equal`, putting NaN at arbitrary positions. `f64::sum()` propagates NaN, making mean/median/p95/p99/min/max all NaN.

**Fix:** Filter out NaN samples before computing stats:
```rust
let clean: Vec<f64> = s.iter().filter(|v| !v.is_nan()).collect();
if clean.is_empty() {
    return Err(Error::Internal("all benchmark samples were NaN".to_string()));
}
```
Compute all statistics from `clean` instead of `s`.

### 2.4 Filter NaN values in top_k_indices
**File:** `cli/src/runner.rs`, line 402

**Problem:** Same `partial_cmp`/NaN pattern in sorting. NaN output values end up at arbitrary rank positions.

**Fix:** Filter out NaN entries before sorting:
```rust
let mut indexed: Vec<(usize, f32)> = v.iter().enumerate()
    .filter(|(_, &val)| !val.is_nan())
    .map(|(i, &val)| (i, val))
    .collect();
```
If all values are NaN, return an empty top-k vector.

### 2.5 Fix median calculation for even-length arrays
**File:** `cli/src/runner.rs`, line 284

**Problem:** `let median = s[n / 2]` returns the upper-middle element for even-length arrays, introducing an upward bias.

**Fix:**
```rust
let median = if n % 2 == 0 {
    (s[n / 2 - 1] + s[n / 2]) / 2.0
} else {
    s[n / 2]
};
```

### 2.6 Propagate serialization errors instead of swallowing them
**File:** `cli/src/main.rs`, lines 228 and 270

**Problem:** `serde_json::to_string_pretty(&v).unwrap_or_else(|_| "{}".into())` — if serialization fails, the user sees an empty JSON object with no error indication.

**Fix:** Replace with proper error propagation:
```rust
println!("{}", serde_json::to_string_pretty(&v).map_err(Error::Parse)?);
```
And add `Parse` as a variant handling in the `dispatch` match.

### 2.7 Fix shape product overflow check
**File:** `cli/src/runner.rs`, line 369

**Problem:** `shape.iter().product()` uses `i64` which can overflow for large shapes, silently wrapping.

**Fix:** Use checked multiplication:
```rust
let expected: Option<i64> = shape.iter().try_fold(1i64, |acc, &d| acc.checked_mul(d));
let expected = expected.ok_or_else(|| Error::Parse("shape dimensions overflow i64".to_string()))?;
```

### 2.8 Validate --top, --runs, --warmup are non-zero
**File:** `cli/src/main.rs`, lines 58-59, 87-92

**Fix:** Add clap value parsers:
```rust
#[arg(value_parser = clap::value_parser!(u32).range(1..))]
--top: u32,

#[arg(value_parser = clap::value_parser!(u32).range(1..))]
--runs: u32,

#[arg(value_parser = clap::value_parser!(u32).range(1..))]
--warmup: u32,
```

### 2.9 Fix unused libc dependency and misleading comment
**File:** `cli/Cargo.toml`, lines 24-27 and 33

**Fix:** Remove the `libc = "0.2"` dependency (it's not used anywhere in the code — the code uses `crucible_free_array` from C, not `libc::free`). Remove or update the comment about `libc`/`malloc` parity to reference `crucible_free_array` instead.

### 2.10 Fix _model_dir error message
**File:** `cli/src/runner.rs`, line 415

**Fix:** Include the model file's directory in the diagnostic message so users know where to place `libcrucible.so`.

### 2.11 Provide default error detail message
**File:** `cli/src/runner.rs`, lines 144-149

**Fix:** When `crucible_last_error()` returns null, use a meaningful default:
```rust
let detail = unsafe {
    let p = crucible_last_error();
    if p.is_null() { "(no detail available from engine)".to_string() } else {
        CStr::from_ptr(p).to_string_lossy().into_owned()
    }
};
```

### 2.12 Add negative dimension check in tensor parsing
**File:** `cli/src/runner.rs`, lines 354-374

**Fix:** After parsing the shape array, validate each dimension:
```rust
for &d in &shape {
    if d < 0 {
        return Err(Error::Parse(format!("negative dimension in shape: {}", d)));
    }
}
```

### 2.13 Remove vestigial dead-code suppression
**File:** `cli/src/main.rs`, lines 120-128

**Fix:** Remove the `let _ = [Status::...]` array — all variants ARE used in the match statement.

---

## Section 3: Server (Python/FastAPI)

### 3.1 Add error handling to _model_dir
**File:** `server/main.py`, line 100

**Fix:** Wrap `mkdir()` in try/except:
```python
try:
    d.mkdir(parents=True, exist_ok=True)
except PermissionError:
    raise HTTPException(500, f"Cannot create model directory at {d}: permission denied")
```

### 3.2 Fix temp file leak in _register_model
**File:** `server/main.py`, lines 182-190

**Fix:** Wrap in try/except with cleanup:
```python
def _register_model(model_id: str, target: Path) -> None:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=_model_dir(), suffix=".onnx", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        shutil.move(str(target), str(tmp_path))
        _MODEL_REGISTRY[model_id] = tmp_path
    except Exception:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
```

### 3.3 Fix silent numpy fallback — don't claim C++ engine
**File:** `server/main.py`, lines 205-223, 378-383; `server/schemas.py`

**Fix:**
- When `BACKEND == "numpy-fallback"`, set the response field to `engine: "crucible-fallback"` instead of `"crucible-cpp"`.
- Add a `backend` field to `/health` response showing the active backend.
- Update the test at `server/tests/test_api.py:381` to assert the correct engine name based on backend.

### 3.4 Remove model_path from API response
**File:** `server/main.py`, lines 378-383; `server/schemas.py`, lines 52-55

**Fix:** Remove `model_path: str` from `ConvertResponse`. Clients should use `model_id` for subsequent requests. If they need the path, they can infer it or use an admin endpoint.

### 3.5 Fix ImportError swallowing in test_bindings
**File:** `server/test_bindings.py`, lines 61-65

**Fix:** Log the exception before continuing:
```python
except ImportError as e:
    print(f"[debug] crucible_py import failed from {path}: {e}")
    continue
```

### 3.6 Fix _safe_torch_load misleading docstring
**File:** `server/converter.py`, lines 111-138

**Fix:** Rewrite the docstring to accurately describe that the function DOES deserialize, but only with `weights_only=True` for safety.

### 3.7 Add path containment check to accept_onnx_upload
**File:** `server/converter.py`, lines 65-108

**Fix:** At the start of the function, validate:
```python
output_path = Path(output_path).resolve()
if not output_path.is_relative_to(_model_dir()):
    raise ValueError(f"output_path must be within model directory: {output_path}")
```

### 3.8 Add validators for inf/nan in InferRequest
**File:** `server/main.py`, line 425; `server/schemas.py`, line 82

**Fix:** Add a Pydantic field validator:
```python
from pydantic import field_validator
import math

@field_validator('input')
@classmethod
def reject_special_floats(cls, v):
    for x in v:
        if math.isnan(x) or math.isinf(x):
            raise ValueError(f"input contains non-finite value: {x}")
    return v
```

### 3.9 Fix floating-point type docstring
**File:** `server/schemas.py`, lines 110-113

**Fix:** Change docstring from "Flattened float32 output" to "Flattened float values serialized as JSON numbers (double precision)."

### 3.10 Fix Pydantic protected_namespaces
**File:** `server/schemas.py`, line 39

**Fix:** Change `protected_namespaces=()` to `protected_namespaces=('model_',)` since no request model uses `model_` prefixed fields.

---

## Section 4: WASM

### 4.1 Fix softmax division by zero
**File:** `wasm/src/lib.rs`, line 579

**Fix:** Guard against sum == 0.0:
```rust
if sum == 0.0 {
    let uniform = 1.0 / axis_dim as f32;
    for j in 0..axis_dim {
        data[start + j] = uniform;
    }
} else {
    for j in 0..axis_dim {
        data[start + j] = exps[j] / sum;
    }
}
```

### 4.2 Fix dimension product overflow
**File:** `wasm/src/lib.rs`, lines 373-374

**Fix:** Use checked multiplication:
```rust
let expected_len: usize = tp.dims.iter().try_fold(1usize, |acc, &d| {
    acc.checked_mul(d as usize).ok_or(ParseError::Overflow)
}).map_err(|_| ParseError::Overflow)?;
```

### 4.3 Replace String::from_utf8_lossy with strict validation
**File:** `wasm/src/lib.rs`, lines 157, 216, 219, 222, 225, 275

**Fix:** Replace `String::from_utf8_lossy()` with `String::from_utf8()` and return a proper error if invalid UTF-8 is found in ONNX string fields (node names, attribute names, etc.):
```rust
String::from_utf8(bytes).map_err(|_| ParseError::InvalidUtf8)?
```

### 4.4 Fix WASM test paths
**File:** `wasm/src/lib.rs`, lines 832, 892

**Fix:** The `std::fs::read` calls with relative paths will fail when cross-compiled for WASM. Wrap these tests in `#[cfg(not(target_arch = "wasm32"))]` or use the WASM-compatible test infrastructure (inject fixture bytes at test runtime rather than reading from disk).

---

## Section 5: Web Frontend

### 5.1 Wire up the Playground to real WASM inference
**File:** `web/src/routes/playground.tsx`, lines 76-91

**Problem:** `run()` computes a fake sigmoid with hardcoded weights. The WASM module is never called.

**Fix:** Import and call the real WASM inference function. The `run()` function should:
1. Validate the shape and inputs
2. Call the WASM module's inference function (from `crucible-wasm.ts`)
3. Display the actual latency and results from WASM
4. If WASM is unavailable, show an error state rather than fabricated results

If the WASM inference function doesn't exist yet, implement it in `crucible-wasm.ts` using the existing `init()` and session APIs.

### 5.2 Implement real file loading in the drop zone
**File:** `web/src/routes/playground.tsx`, lines 93-98

**Fix:** In `onDrop`, read the file bytes with `FileReader`:
```typescript
const reader = new FileReader();
reader.onload = (e) => {
    const bytes = new Uint8Array(e.target?.result as ArrayBuffer);
    // Pass bytes to WASM model loader
};
reader.readAsArrayBuffer(f);
```

### 5.3 Fix heuristicScore to be deterministic
**File:** `web/src/routes/fraud.tsx`, line 99

**Fix:** Remove `Math.random()`. The fallback should compute a deterministic score based on the input values. For example, use a simple weighted sum with sigmoid:
```typescript
function heuristicScore(v: number[]): number {
    const weights = [0.1, -0.05, 0.3, 0.02, -0.08, 0.15, 0.05];
    const sum = v.reduce((a, b, i) => a + b * weights[i], 0);
    return 1 / (1 + Math.exp(-sum));
}
```
Better yet, if WASM fails to load, surface the error to the user instead of silently substituting a fake result.

### 5.4 Fix NaN in shape input
**File:** `web/src/routes/playground.tsx`, lines 100-102

**Fix:** Validate the input value before using it:
```typescript
const updateShape = (i: number, v: number) => {
    setShape((s) => s.map((x, idx) => {
        if (idx !== i) return x;
        const parsed = parseInt(e.target.value);
        return isNaN(parsed) || parsed < 1 ? 1 : parsed;
    }));
};
```

### 5.5 Add range validation on fraud form fields
**File:** `web/src/routes/fraud.tsx`, lines 200-246

**Fix:** Add `min` and `max` attributes to each `<input type="number">`:
- `amount`: min="0"
- `oldBalanceOrig`: min="0"
- `newBalanceOrig`: min="0"
- `oldBalanceDest`: min="0"
- `newBalanceDest`: min="0"
Also add `step="0.01"` for any decimal values.

### 5.6 Fix module-level data computation in Benchmark
**File:** `web/src/routes/benchmark.tsx`, lines 36-47

**Fix:** Move `getBenchmarkResults()` and `getChartData()` calls inside the component body (useEffect or direct call in render), not at module scope. Also add null/undefined guards:
```typescript
const _bdata = getBenchmarkResults();
const _crucible = _bdata?.results?.find((r) => r.engine === "crucible");
if (!_crucible) return <div>No benchmark data available</div>;
```

### 5.7 Add keyboard accessibility to drop zone
**File:** `web/src/routes/playground.tsx`, lines 123-143

**Fix:** Add:
- `role="button"` or `role="textbox"`
- `tabIndex={0}`
- `aria-label="Drop an ONNX model file here or click to browse"`
- An `onKeyDown` handler for Enter/Space that triggers a hidden `<input type="file">`
- An `onClick` that triggers the same file input

### 5.8 Add aria-expanded to graph node toggles
**File:** `web/src/routes/playground.tsx`, lines 305-343

**Fix:** Add `aria-expanded={expanded === i}` and `aria-controls={`node-detail-${i}`}` to the toggle buttons.

### 5.9 Add aria-current to active nav link
**File:** `web/src/components/crucible/Layout.tsx`, line 47

**Fix:** Add `aria-current="page"` to the `<Link>` component for the active route.

### 5.10 Fix MQL listener cleanup
**File:** `web/src/hooks/use-mobile.tsx`, lines 9, 13, 15

**Fix:** Use the universally supported `addListener`/`removeListener` API:
```typescript
const mql = window.matchMedia(`(max-width: ${MOBILE_BREAKPOINT - 1}px)`);
mql.addListener(onChange);
return () => mql.removeListener(onChange);
```

### 5.11 Add autocomplete="off" to fraud form
**File:** `web/src/routes/fraud.tsx`, lines 200-246

**Fix:** Add `autoComplete="off"` to all financial `<input type="number">` fields.

### 5.12 Fix inline onclick in error page
**File:** `web/src/lib/error-page.ts`, line 24

**Fix:** Replace inline `onclick="location.reload()"` with a React event handler or add a CSP-compatible approach.

---

## Section 6: CI/CD, Docker, and Deployment

### 6.1 Fix CI-Rust workflow trigger
**File:** `.github/workflows/ci-rust.yml`

**Fix:** The `paths` filter references `cli/**` but the CI should trigger on the actual Rust crate files. Change to:
```yaml
paths:
  - 'cli/**'
  - '.github/workflows/ci-rust.yml'
```
Or if the `cli/` directory hasn't been created yet, either create it or remove the path filter entirely.

### 6.2 Fix CMake cache key
**File:** `.github/workflows/ci-engine.yml`, line 56

**Fix:** Add `engine/CMakePresets.json` to the hashFiles:
```yaml
key: cmake-debug-${{ runner.os }}-${{ hashFiles('engine/CMakeLists.txt', 'engine/src/**', 'engine/include/**', 'engine/CMakePresets.json') }
```

### 6.3 Fix Vercel build command
**File:** `vercel.json`, line 2

**Fix:** Remove the broken `cp -r .vercel ../` line and add proper install/build steps:
```json
{
  "installCommand": "cd web && npm install",
  "buildCommand": "cd web && npm run build"
}
```

### 6.4 Fix Render backend service
**File:** `render.yaml`, lines 19-28

**Fix:** Uncomment and configure the `crucible-api` service, or document that it is intentionally disabled with a comment explaining why.

### 6.5 Fix Dockerfile submodule fallback
**File:** `Dockerfile`, lines 58-69

**Fix:** Ensure the fallback clone URLs match `.gitmodules` exactly. Also add `googletest` and `google-benchmark` to the fallback list if tests are enabled:
```dockerfile
# Add missing submodule fallbacks
RUN git clone --depth 1 --branch v1.14.0 https://github.com/google/googletest.git third_party/googletest
```

### 6.6 Add subprocess timeouts to benchmarks
**File:** `benchmarks/run_all.py`, lines 229-236

**Fix:** Add `timeout=120` (seconds) to the `subprocess.run` call and handle `subprocess.TimeoutExpired`:
```python
try:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
except subprocess.TimeoutExpired:
    print(f"Benchmark {engine} timed out after 120s")
    results.append({"engine": engine, "error": "timeout"})
    continue
```

---

## Section 7: C++ Engine Headers

### 7.1 Fix misleading explicit keyword
**File:** `engine/include/crucible/tensor.hpp`, line 47

**Fix:** Remove `explicit` (it's a no-op for multi-arg constructors with defaults) or document why it's there. If the intent is to prevent `Tensor t = {2, 3}` copy-initialization, C++ doesn't support that restriction — use a named constructor or factory function.

### 7.2 Document/fix flatten() behavior for scalar tensors
**File:** `engine/include/crucible/tensor.hpp`, line 82

**Fix:** Add explicit handling for rank-0 tensors. Either reject them with a clear error, or define `flatten()` on a scalar to return a rank-1 tensor of size 1. Add a doc comment explaining the behavior.

### 7.3 Fix exception types in at()
**File:** `engine/include/crucible/tensor.hpp`, lines 68-69

**Fix:** Use `std::invalid_argument` for rank mismatch and `std::out_of_range` for index value overflow. Update tests accordingly.

### 7.4 Fix boolean transpose flags in gemm()
**File:** `engine/include/crucible/ops/linear.hpp`, lines 30-32

**Fix:** Change `int transA = 0` to `bool transA = false` and `int transB = 0` to `bool transB = false`.

---

## Section 8: Documentation and Hygiene

### 8.1 Fix .gitignore
**File:** `.gitignore` (at repo root)

**Fix:** Ensure it includes standard Rust/Python/Node ignores: `target/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`, `node_modules/`, `.vercel/`, `.DS_Store`, `*.swp`, `.env`, `.env.local`.

### 8.2 Add .env.example
**File:** Create `.env.example`

**Fix:** Create with at least:
```
CRUCIBLE_MODEL_DIR=/tmp/models
CRUCIBLE_ENGINE_PATH=/usr/local/lib/libcrucible.so
CRUCIBLE_API_KEY=change-me-in-production
CRUCIBLE_MAX_MODEL_SIZE_MB=100
```

### 8.3 Fix percentile formula in benchmarks
**Files:** `benchmarks/bench_crucible.py`, `benchmarks/bench_onnxruntime.py`, `benchmarks/bench_pytorch.py`

**Fix:** Replace the nearest-rank formula with linear interpolation (standard percentile):
```python
def percentile(sorted_data, p):
    n = len(sorted_data)
    if n == 1:
        return sorted_data[0]
    k = (n - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    d0 = sorted_data[f] * (c - k)
    d1 = sorted_data[c] * (k - f)
    return d0 + d1
```

---

## Execution Order

Work through the sections in this order (root causes first):

1. **Section 1** (Models) — fixes the data leak and ONNX export crash
2. **Section 2** (Rust CLI) — fixes safety bugs in the most complete component
3. **Section 3** (Server) — fixes crashes, leaks, and security issues
4. **Section 4** (WASM) — fixes correctness bugs in the runtime
5. **Section 5** (Web) — fixes fake inference and accessibility issues
6. **Section 6** (CI/CD) — fixes deployment pipelines
7. **Section 7** (C++ headers) — fixes API contract issues
8. **Section 8** (Hygiene) — docs, gitignore, env templates

After completing all fixes, run `cargo check` in `cli/`, `python -m pytest` in `server/`, and `npm run type-check` in `web/` to verify nothing is broken.
