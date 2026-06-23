# benchmarks/results/

This directory holds the captured output of the `crucible_benchmarks`
executable, one file per "story" we want to tell.

## How a result file is produced

After configuring and building both Debug and Release presets:

```bash
# Configure + build
cmake --preset debug   -S engine
cmake --preset release -S engine
cmake --build build/debug   --target crucible_benchmarks
cmake --build build/release --target crucible_benchmarks

# Run each, capturing JSON to a file
./build/debug/engine/benchmarks/crucible_benchmarks \
    --benchmark_format=console \
    > /dev/null                                                     # warm cache
./build/debug/engine/benchmarks/crucible_benchmarks \
    --benchmark_format=json \
    --benchmark_out=benchmarks/results/matmul_debug_vs_release_debug.json

./build/release/engine/benchmarks/crucible_benchmarks \
    --benchmark_format=json \
    --benchmark_out=benchmarks/results/matmul_debug_vs_release_release.json
```

Then merge them into the side-by-side comparison file
`matmul_debug_vs_release.json` with `python benchmarks/results/merge.py`
(see Issue #11 commit for the script).

## File naming convention

`<topic>[_<build>].json` — the per-build files use Google's native
JSON output (one record per benchmark). The merged file is a
flat list of `{name, build, time_unit, real_time, cpu_time, ...}`
records so the reader can compare Debug vs Release row-by-row
without parsing Google's nested format twice.