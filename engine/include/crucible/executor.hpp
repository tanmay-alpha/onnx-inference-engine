// Graph executor — Issue #9.
//
// This is the runtime heart of Crucible. It takes a parsed `Model`
// (graph + initializers + IO names) plus a map of float32 input
// tensors and walks the graph in dependency order, dispatching each
// `GraphNode` to the operator implementation that owns its `op_type`.
//
// Two responsibilities split cleanly across the two public methods:
//
//   1. topological_sort — pure functional. Given the node list, return
//      a copy reordered so every node appears after every node it
//      depends on. Kahn's BFS; throws on cycles or unknown edges.
//
//   2. run_inference     — stateful. Holds a `tensor_map` of every
//      known tensor name to its materialised value, materialises each
//      node in topological order, and returns the value bound to the
//      model's first output.
//
// Design notes:
//
//   * We don't compile the graph ahead of time. Issue #9 is the
//     interpreter; Issue #12 (#10 in older drafts) will fuse and JIT
//     once we know which fusion patterns actually fire. The shape of
//     the API is deliberately close to how an ONNX Runtime session
//     looks: load_model() once, run_inference() many times. Issue #11
//     will add a Session wrapper around that.
//
//   * Dispatch is an if/else chain on `op_type`, not a std::function
//     table. 13 ops with 1-3 args each doesn't benefit from type
//     erasure; the chain is the readable form. When the count crosses
//     ~30 we'll switch to a flat dispatch table — for now, clarity wins.
//
//   * `tensor_map` is a `std::unordered_map<std::string, Tensor>`.
//     The Tensor is moved (not copied) when a node's output is stored
//     — the operator functions return by value, and the move
//     constructor is the cheap path.

#pragma once

#include "crucible/onnx_parser.hpp"
#include "crucible/tensor.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace crucible {

// Run a parsed ONNX model and produce the value of its first listed
// output.
//
//   model  : a parsed graph (see onnx_parser.hpp)
//   inputs : map from `model.input_names[i]` to the float32 tensor
//            value to feed. The keys are matched against
//            `model.input_names`; an unknown key throws
//            std::invalid_argument, and a missing key throws
//            std::invalid_argument with a clear "no input for X" message.
//
// Throws std::invalid_argument on:
//   * missing / extra input names
//   * unknown op_type
//   * shape / rank mismatch in any operator (the operators themselves
//     throw; we just propagate)
// Throws std::runtime_error on:
//   * graph with a cycle (topological_sort can't find a complete order)
//   * a node that consumes a tensor which is neither an initializer
//     nor a prior node's output (dangling edge)
Tensor run_inference(const Model& model,
                     const std::unordered_map<std::string, Tensor>& inputs);

// Return a copy of `nodes` reordered so that every node appears after
// every node it depends on. "Depends on" means: the node's `inputs`
// list contains a name that is some other node's `outputs` entry.
//
//   nodes : the graph's node list, in any order
//
// Throws std::runtime_error on a cycle or a self-loop. The algorithm
// is Kahn's BFS — in-degree of every node, push the in-degree-0 set,
// pop, decrement consumers' in-degrees. Linear in |V|+|E|, which is
// what every modern ONNX runtime uses for the same problem.
std::vector<GraphNode> topological_sort(const std::vector<GraphNode>& nodes);

}  // namespace crucible
