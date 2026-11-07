// Graph executor.
//
// Takes a parsed `Model` (graph + initializers + IO metadata) plus a map of
// float32 input tensors, walks the graph in dependency order and dispatches
// each `GraphNode` to the operator that owns its `op_type`.
//
// Design notes:
//
//   * Interpreter, no graph compilation or fusion: load_model() once,
//     run_inference() many times. `Model` is never mutated by a run, so
//     concurrent runs on the same Model are safe.
//
//   * Initializers and caller inputs are read in place; only node outputs
//     are owned by the run, and each intermediate is released right after
//     its last consumer executes.
//
//   * Anything Crucible cannot compute exactly as the ONNX spec defines it
//     (unknown op, foreign domain, dilations, asymmetric pads, ceil_mode,
//     non-float inputs, ...) is rejected with a std::runtime_error naming the
//     node and the offending attribute — never approximated silently.

#pragma once

#include "crucible/onnx_parser.hpp"
#include "crucible/tensor.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace crucible {

// Run a parsed ONNX model and return the value of its first listed output.
//
// Throws std::invalid_argument on:
//   * a missing graph input, a non-float32 graph input, or an input whose
//     shape contradicts the model's declared static dimensions
//   * shape / rank mismatch inside an operator
// Throws std::runtime_error on:
//   * unsupported operator, domain or attribute value
//   * graph with a cycle
//   * a node consuming a tensor that is neither a graph input, an
//     initializer, nor a prior node's output
Tensor run_inference(const Model& model,
                     const std::unordered_map<std::string, Tensor>& inputs);

// Static support check: one human-readable reason per problem that would
// make run_inference fail regardless of input values (unsupported op /
// attribute / dtype, cycle, no outputs). Empty = the model can run.
std::vector<std::string> check_model(const Model& model);

// The ONNX op_types (default domain) the executor implements, sorted.
const std::vector<std::string>& supported_ops();

// Return a copy of `nodes` reordered so that every node appears after every
// node it depends on (Kahn's algorithm). Throws std::runtime_error on a
// cycle or self-loop.
std::vector<GraphNode> topological_sort(const std::vector<GraphNode>& nodes);

}  // namespace crucible
