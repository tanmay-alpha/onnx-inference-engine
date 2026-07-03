//! Crucible CLI — Issue #15.
//!
//! Four subcommands backed by the C ABI in `engine/include/crucible/c_api.h`:
//!
//!   crucible run        --model mobilenet_v2.onnx --input input.json
//!   crucible benchmark  --model mobilenet_v2.onnx --input input.json
//!   crucible validate   --model mobilenet_v2.onnx
//!   crucible info       --model mobilenet_v2.onnx
//!
//! The first three are the AC-required surface for the plan; `info`
//! is a small but useful command for sanity-checking a model file
//! without writing it through inference.
//!
//! Exit codes follow the conventional CLI layout:
//!   0 — success
//!   1 — user error (bad args, missing file, model parse failure)
//!   2 — engine error (runtime failure, unsupported operator)
//!
//! Output mode is selected with --json. Without it, the CLI prints
//! human-readable text on stdout. Errors always go to stderr.

mod runner;
mod formatter;

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use clap::{Parser, Subcommand, ValueHint};

#[derive(Debug, Parser)]
#[command(
    name        = "crucible",
    about       = "Crucible ONNX inference engine — command-line front-end",
    long_about  = "Crucible is a C++ ONNX inference engine. This CLI loads a model via \
                   the engine's C ABI, runs inference, and prints results. See \
                   `crucible <command> --help` for command-specific options.",
    version     = env!("CARGO_PKG_VERSION"),
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Load a model and run one inference, printing the first output.
    Run {
        /// Path to the .onnx model file.
        #[arg(short, long, value_name = "FILE", value_hint = ValueHint::FilePath)]
        model: PathBuf,

        /// Path to a JSON tensor file with {"shape":[], "data":[]}.
        #[arg(short, long, value_name = "FILE", value_hint = ValueHint::FilePath)]
        input: PathBuf,

        /// Number of top predictions to print (default 5).
        #[arg(long, value_name = "N", default_value_t = 5)]
        top: usize,

        /// Path to a JSON file with class labels (one per line, in
        /// the same order as the output tensor's class axis).
        #[arg(long, value_name = "FILE", value_hint = ValueHint::FilePath)]
        labels: Option<PathBuf>,

        /// Instead of top-k, print the full output tensor (shape +
        /// first N values). Useful for non-classification models.
        #[arg(long, value_name = "N")]
        print_output: Option<usize>,

        /// Emit machine-readable JSON on stdout.
        #[arg(long)]
        json: bool,
    },

    /// Run a model N times and report latency statistics.
    Benchmark {
        /// Path to the .onnx model file.
        #[arg(short, long, value_name = "FILE", value_hint = ValueHint::FilePath)]
        model: PathBuf,

        /// Path to a JSON tensor file with {"shape":[], "data":[]}.
        #[arg(short, long, value_name = "FILE", value_hint = ValueHint::FilePath)]
        input: PathBuf,

        /// Number of timed runs after warmup.
        #[arg(long, value_name = "N", default_value_t = 100)]
        runs: usize,

        /// Number of un-timed warmup runs.
        #[arg(long, value_name = "N", default_value_t = 10)]
        warmup: usize,

        /// Emit machine-readable JSON on stdout.
        #[arg(long)]
        json: bool,
    },

    /// Load a model, fail if it doesn't parse, and exit.
    Validate {
        /// Path to the .onnx model file.
        #[arg(short, long, value_name = "FILE", value_hint = ValueHint::FilePath)]
        model: PathBuf,
    },

    /// Print static model metadata (IO names, node count).
    Info {
        /// Path to the .onnx model file.
        #[arg(short, long, value_name = "FILE", value_hint = ValueHint::FilePath)]
        model: PathBuf,

        /// Emit machine-readable JSON on stdout.
        #[arg(long)]
        json: bool,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match dispatch(cli) {
        Ok(())  => ExitCode::SUCCESS,
        Err(e)  => {
            // User-visible errors always go to stderr with a stable
            // "error:" prefix so log scrapers can find them.
            eprintln!("error: {e}");
            // Map our error categories to conventional exit codes.
            // CrucibleError::LibraryUnavailable is a user error
            // (build the engine); CrucibleError::Runtime and
            // CrucibleError::Internal are engine errors.
            match e {
                runner::CrucibleError::InvalidArgument(_)
                | runner::CrucibleError::Io(_)
                | runner::CrucibleError::Parse(_)
                | runner::CrucibleError::Fs(_)
                | runner::CrucibleError::LibraryUnavailable(_) => ExitCode::from(1),
                runner::CrucibleError::Runtime(_)
                | runner::CrucibleError::Unsupported(_)
                | runner::CrucibleError::Internal(_)         => ExitCode::from(2),
            }
        }
    }
}

fn dispatch(cli: Cli) -> Result<(), runner::CrucibleError> {
    match cli.command {
        Command::Run { model, input, top, labels, print_output, json } => {
            cmd_run(&model, &input, top, labels.as_deref(),
                    print_output, json)
        }
        Command::Benchmark { model, input, runs, warmup, json } => {
            cmd_bench(&model, &input, runs, warmup, json)
        }
        Command::Validate { model }              => cmd_validate(&model),
        Command::Info { model, json }            => cmd_info(&model, json),
    }
}

// ---------------------------------------------------------------------------
// run
// ---------------------------------------------------------------------------

fn cmd_run(
    model_path: &Path,
    input_path: &Path,
    top: usize,
    labels_path: Option<&Path>,
    print_output: Option<usize>,
    json: bool,
) -> Result<(), runner::CrucibleError> {
    let model = runner::Model::load(model_path)?;
    let input = runner::Tensor::from_json_file(input_path)?;
    let outputs = model.run(&[input])?;
    // The C side only fills the first output for now; later we'll
    // surface all of them once Issue #11's Session lands.
    let out = outputs.into_iter().next().unwrap_or_else(runner::Tensor::empty);

    let labels = match labels_path {
        Some(p) => Some(load_labels(p)?),
        None    => None,
    };

    if json {
        let v = if let Some(max_n) = print_output {
            // Full output mode: shape + first N values.
            let first_n: Vec<serde_json::Value> = out.data
                .iter()
                .take(max_n)
                .map(|f| serde_json::Value::from(*f as f64))
                .collect();
            serde_json::json!({
                "mode":         "raw",
                "shape":        out.shape,
                "data_first_n": first_n,
                "n_truncated":  out.data.len().saturating_sub(max_n),
            })
        } else {
            // Top-k mode: rank + class_idx + name + probability.
            let topk = runner::top_k_indices(&out.data, top);
            let mut arr = Vec::with_capacity(topk.len());
            for (rank_idx, (class_idx, prob)) in topk.iter().enumerate() {
                let name = labels
                    .as_ref()
                    .and_then(|c| c.get(*class_idx).cloned())
                    .unwrap_or_else(|| format!("class#{class_idx}"));
                arr.push(serde_json::json!({
                    "rank":        rank_idx + 1,
                    "class_idx":   class_idx,
                    "name":        name,
                    "probability": prob,
                }));
            }
            serde_json::json!({
                "mode":  "top_k",
                "shape": out.shape,
                "top":   arr,
            })
        };
        println!("{}", serde_json::to_string_pretty(&v).unwrap_or_else(|_| "{}".into()));
    } else {
        // Text mode.
        if let Some(max_n) = print_output {
            print!("{}", formatter::format_tensor_summary(&out, max_n));
        } else {
            print!("{}", formatter::format_top_k_predictions(&out, labels.as_deref(), top));
        }
        formatter::flush();
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// benchmark
// ---------------------------------------------------------------------------

fn cmd_bench(
    model_path: &Path,
    input_path: &Path,
    runs: usize,
    warmup: usize,
    json: bool,
) -> Result<(), runner::CrucibleError> {
    let model = runner::Model::load(model_path)?;
    let input = runner::Tensor::from_json_file(input_path)?;

    // Warmup.
    for _ in 0..warmup {
        let _ = model.run(&[input.clone()])?;
    }
    // Timed.
    let mut samples_ms: Vec<f64> = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t0 = Instant::now();
        let _ = model.run(&[input.clone()])?;
        let dt = t0.elapsed();
        samples_ms.push(dt.as_secs_f64() * 1000.0);
    }
    let (mean, median, p95, p99, mn, mx) = compute_stats(&samples_ms);
    if json {
        let v = formatter::format_bench_json(runs, mean, median, p95, p99, mn, mx);
        println!("{}", serde_json::to_string_pretty(&v).unwrap_or_else(|_| "{}".into()));
    } else {
        print!("{}", formatter::format_bench_text(runs, mean, median, p95, p99, mn, mx));
        formatter::flush();
    }
    Ok(())
}

fn compute_stats(samples: &[f64]) -> (f64, f64, f64, f64, f64, f64) {
    if samples.is_empty() { return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0); }
    let mut s: Vec<f64> = samples.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = s.len();
    let mean   = s.iter().sum::<f64>() / n as f64;
    let median = s[n / 2];
    let p95    = s[((n as f64 * 0.95).floor() as usize).min(n - 1)];
    let p99    = s[((n as f64 * 0.99).floor() as usize).min(n - 1)];
    let mn     = s[0];
    let mx     = s[n - 1];
    (mean, median, p95, p99, mn, mx)
}

// ---------------------------------------------------------------------------
// validate
// ---------------------------------------------------------------------------

fn cmd_validate(model_path: &Path) -> Result<(), runner::CrucibleError> {
    let info = runner::validate_model(model_path)?;
    print!("{}", formatter::format_validate_text(model_path, &info));
    formatter::flush();
    Ok(())
}

// ---------------------------------------------------------------------------
// info
// ---------------------------------------------------------------------------

fn cmd_info(model_path: &Path, json: bool) -> Result<(), runner::CrucibleError> {
    let model = runner::Model::load(model_path)?;
    let info  = model.info()?;
    if json {
        let v = formatter::format_model_info_json(&info, model_path);
        println!("{}", serde_json::to_string_pretty(&v).unwrap_or_else(|_| "{}".into()));
    } else {
        print!("{}", formatter::format_model_info_text(&info, model_path));
        formatter::flush();
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Load class labels from a file, one label per line. Empty lines
/// and `# comment` lines are skipped. Whitespace is trimmed.
fn load_labels(path: &Path) -> Result<Vec<String>, runner::CrucibleError> {
    let s = std::fs::read_to_string(path)?;
    let mut out = Vec::new();
    for line in s.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') { continue; }
        out.push(t.to_string());
    }
    Ok(out)
}
