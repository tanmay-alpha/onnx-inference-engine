// Graph executor — see executor.hpp for the public API.
//
// This file is the runtime heart of Crucible. It owns two pieces:
//
//   1. `topological_sort` — pure functional. Kahn's BFS over the
//      node list, treating "input name = X" and "some prior node
//      outputs X" as an inter-node edge. Returns the sorted copy.
//
//   2. `run_inference`     — stateful. Builds a `tensor_map` of every
//      named tensor (initializers + graph inputs + intermediate node
//      outputs) and walks the sorted node list, dispatching each one
//      to the right operator. Returns the tensor bound to the model's
//      first output.
//
// Concurrency / threading:
//   The executor is a pure interpreter on this issue; nothing inside
//   run_inference mutates the input `Model`, and the tensor_map is a
//   local variable. A future issue will own the parallel-execution
//   decision (independent nodes, shared-nothing per-thread maps).
//
// On the dispatch table:
//   Each `op_type` is a string match. We use an if/else chain instead
//   of a std::function map because (a) at 13 ops the chain is the
//   readable form, (b) we don't need type erasure since each branch
//   binds to its operator's static signature, and (c) the compiler
//   can hoist the string comparisons out into a jump table. At 30+
//   ops we'll revisit.

#include "crucible/executor.hpp"

#include "crucible/ops/activations.hpp"
#include "crucible/ops/conv2d.hpp"
#include "crucible/ops/linear.hpp"
#include "crucible/ops/norm.hpp"
#include "crucible/ops/pooling.hpp"

#include <climits>
#include <cmath>
#include <cstdint>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace crucible {

// ====================================================================
// topological_sort — Kahn's BFS
// ====================================================================

std::vector<GraphNode> topological_sort(const std::vector<GraphNode>& nodes) {
    // Build a name -> producer map. For every (input_name, target_node)
    // pair, look up whether some prior node produces that name as an
    // output. If so, target_node depends on that producer.
    //
    // We allow the input node list to NOT be in topological order —
    // the algorithm below discovers the actual dependency graph from
    // the input/output name wiring.
    std::unordered_map<std::string, std::size_t> producer;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        for (const auto& out_name : nodes[i].outputs) {
            // If two nodes share an output name, we treat the first
            // one as the producer and ignore the second — this should
            // not happen in a well-formed ONNX model, but emitting a
            // duplicate name is much easier to debug than crashing.
            if (producer.find(out_name) == producer.end()) {
                producer[out_name] = i;
            }
        }
    }

    // in_degree[i] = number of node-i's inputs that come from some
    // other node (vs. an initializer / graph input).
    std::vector<int> in_degree(nodes.size(), 0);
    std::vector<std::vector<std::size_t>> consumers(nodes.size());
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        for (const auto& in_name : nodes[i].inputs) {
            auto it = producer.find(in_name);
            if (it == producer.end()) continue;   // comes from an initializer/input
            const std::size_t p = it->second;
            if (p == i) {
                throw std::runtime_error(
                    "topological_sort: self-loop on node '" +
                    nodes[i].name + "' (op " + nodes[i].op_type + ")");
            }
            ++in_degree[i];
            consumers[p].push_back(i);
        }
    }

    // Seed the queue with every node that has no node-to-node
    // dependencies. std::queue gives FIFO order which is what Kahn's
    // algorithm wants.
    std::queue<std::size_t> ready;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        if (in_degree[i] == 0) ready.push(i);
    }

    std::vector<GraphNode> sorted;
    sorted.reserve(nodes.size());
    while (!ready.empty()) {
        const std::size_t i = ready.front();
        ready.pop();
        sorted.push_back(nodes[i]);
        for (std::size_t c : consumers[i]) {
            if (--in_degree[c] == 0) ready.push(c);
        }
    }

    if (sorted.size() != nodes.size()) {
        throw std::runtime_error(
            "topological_sort: cycle detected (sorted " +
            std::to_string(sorted.size()) + " of " +
            std::to_string(nodes.size()) + " nodes)");
    }
    return sorted;
}

// ====================================================================
// Internal helpers for run_inference
// ====================================================================

namespace {

// Pull a 1-D float attribute out of `attrs` as `int`. ONNX shapes
// reach us as int64 attributes, but our activation operator accepts
// them as float in the `unordered_map<string, float>` map. We coerce
// here once, at the dispatch boundary.
int attr_int(const std::unordered_map<std::string, float>& attrs,
             const std::string& name, int fallback) {
    auto it = attrs.find(name);
    return (it == attrs.end()) ? fallback : static_cast<int>(it->second);
}

float attr_float(const std::unordered_map<std::string, float>& attrs,
                 const std::string& name, float fallback) {
    auto it = attrs.find(name);
    if (it == attrs.end()) return fallback;
    if (!std::isfinite(it->second)) {
        throw std::invalid_argument(
            "ONNX attribute '" + name + "' is not finite (got " +
            std::to_string(it->second) + ")");
    }
    return it->second;
}

// Convert `model.attributes` (the `Attribute` union from onnx_parser.hpp)
// into the flat `unordered_map<string, float>` that the activation ops
// accept. We pass int-valued attributes through the float channel as
// their integer value cast to float — activations only consume axis
// and approximate-constant float attrs (none of which overflow float32).
std::unordered_map<std::string, float> flatten_attrs(
    const std::unordered_map<std::string, Attribute>& src) {
    std::unordered_map<std::string, float> out;
    out.reserve(src.size());
    for (const auto& [k, v] : src) {
        switch (v.type) {
            case Attribute::Type::Float: out[k] = v.f; break;
            case Attribute::Type::Int:   out[k] = static_cast<float>(v.i); break;
            // FloatArray / IntArray / String aren't consumed by any
            // operator in the dispatch table; skip silently. A future
            // issue will need to extend the activation ops if it adds
            // ops that read arrays (e.g. Slice, Gather).
            case Attribute::Type::FloatArray:
            case Attribute::Type::IntArray:
            case Attribute::Type::String: break;
        }
    }
    return out;
}

// Concat along axis 0 (the only axis the executor-level Concat needs
// for MobileNetV2's stem-free backbone). Each input must have the
// same shape; the output shape is the same as each input except
// axis 0 is the sum of all inputs' axis 0.
//
// We could generalise to arbitrary axis by permute+reshape, but
// MobileNetV2's only concat is along the channel axis and that case
// is what tests will exercise once Issue #11 loads the real model.
// Element-wise broadcast Add is also implemented here for the same
// reason — both are tiny, only consumed by the executor, and pulling
// them into their own header just adds files without adding clarity.
Tensor concat_axis0(const std::vector<Tensor>& parts) {
    if (parts.empty()) {
        throw std::invalid_argument("Concat: at least one input required");
    }
    const auto& ref_shape = parts[0].shape();
    const int64_t rank = static_cast<int64_t>(ref_shape.size());
    if (rank == 0) {
        throw std::invalid_argument(
            "Concat: inputs must be at least rank 1, got scalar");
    }
    for (std::size_t i = 1; i < parts.size(); ++i) {
        if (parts[i].shape() != ref_shape) {
            throw std::invalid_argument(
                "Concat: all inputs must share the same shape (axis-0 "
                "concat only — got mismatched shapes)");
        }
    }

    // Output shape: same as input, axis 0 = sum of axes 0.
    std::vector<int64_t> out_shape = ref_shape;
    int64_t axis0_sum = 0;
    for (const auto& p : parts) {
        if (axis0_sum > INT64_MAX - p.shape()[0]) {
            throw std::overflow_error(
                "Concat: axis-0 dimension sum overflows int64");
        }
        axis0_sum += p.shape()[0];
    }
    out_shape[0] = axis0_sum;

    Tensor out(out_shape, 0.0f);
    int64_t offset = 0;
    for (const auto& p : parts) {
        const int64_t rows = p.shape()[0];
        const int64_t per_row = p.size() / rows;
        // Copy per-row contiguous chunks into out at row offset.
        // We use raw offsets because Eigen would add nothing here:
        // this is a memcpy, not a math op.
        std::copy(p.data(),
                  p.data() + rows * per_row,
                  out.data() + offset * per_row);
        offset += rows;
    }
    return out;
}

// Broadcast element-wise Add. Supports the cases that come up in
// MobileNetV2's residual connections:
//   * T + T   same-shape tensors
//   * T + (C,) broadcasting (bias add along the channel axis for NCHW
//     where C is the trailing axis). PyTorch broadcasts (C,) over
//     rank-4 NCHW with the channel axis implicit-last, so we widen
//     the bias by repeating across N, H, W.
//   * T + (1,) or T + scalar — trivial broadcast.
// Anything more exotic (5-D tensors, stride patterns) is out of scope
// for Issue #9; the executor-level Add is a convenience for the
// next issue's MobileNetV2 wiring tests.
Tensor add_broadcast(const Tensor& A, const Tensor& B) {
    // Scalar broadcast (rank-0).
    if (A.shape().empty() || B.shape().empty()) {
        const float b = B.shape().empty() ? B.data()[0] : 0.0f;
        const float a = A.shape().empty() ? A.data()[0] : 0.0f;
        Tensor out({1}, 0.0f);
        out.data()[0] = a + b;
        return out;
    }
    // (C,) bias add to (N, C, H, W) — match the trailing axis.
    // This is the broadcast pattern that MobileNetV2's residual
    // connections and the Conv2D bias path both need. We compute the
    // inner spatial size as (total) / (N*C) and add the matching
    // bias element to every (n, c) slab.
    if (B.shape().size() == 1 && A.shape().size() >= 2 &&
        B.shape()[0] == A.shape().back()) {
        Tensor out(A.shape(), 0.0f);
        const int64_t C = B.shape()[0];
        const int64_t spatial = A.size() / C / A.shape()[0];
        for (int64_t n = 0; n < A.shape()[0]; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                const float bc = B.data()[c];
                const int64_t base = (n * C + c) * spatial;
                for (int64_t k = 0; k < spatial; ++k) {
                    out.data()[base + k] = A.data()[base + k] + bc;
                }
            }
        }
        return out;
    }
    // Same-shape add.
    if (A.shape() == B.shape()) {
        Tensor out(A.shape(), 0.0f);
        for (int64_t i = 0; i < A.size(); ++i) {
            out.data()[i] = A.data()[i] + B.data()[i];
        }
        return out;
    }
    throw std::invalid_argument(
        "Add: unsupported broadcast (A shape [" + [&]{
            std::string s;
            for (auto x : A.shape()) { s += std::to_string(x); s += ","; }
            return s;
        }() + "] B shape [" + [&]{
            std::string s;
            for (auto x : B.shape()) { s += std::to_string(x); s += ","; }
            return s;
        }() + "])");
}

// Look up `name` in the float tensor map; throw a clear error if
// absent. We don't fall back to int_initializers here — that's the
// dispatcher's job (Reshape specifically knows it has an int64 shape
// input).
const Tensor& require_tensor(
    const std::unordered_map<std::string, Tensor>& tm,
    const std::string& name) {
    auto it = tm.find(name);
    if (it == tm.end()) {
        throw std::runtime_error(
            "run_inference: tensor '" + name + "' is neither an "
            "initializer nor a prior node's output");
    }
    return it->second;
}

}  // namespace

// ====================================================================
// run_inference
// ====================================================================

Tensor run_inference(const Model& model,
                     const std::unordered_map<std::string, Tensor>& inputs) {
    // 1. Validate that the caller supplied exactly the inputs the
    //    graph expects. We allow extra keys in `inputs` (silent ignore,
    //    useful when a Session passes all params) but require every
    //    `model.input_names[i]` to be present.
    for (const auto& name : model.input_names) {
        if (model.weights.find(name) != model.weights.end() ||
            model.int_initializers.find(name) != model.int_initializers.end()) {
            continue;
        }
        if (inputs.find(name) == inputs.end()) {
            throw std::invalid_argument(
                "run_inference: no input provided for graph input '" +
                name + "'");
        }
    }

    // 2. Seed tensor_map: weights (initializers) and user-supplied
    //    inputs. int_initializers live in a separate map and are
    //    resolved on-demand by Reshape.
    std::unordered_map<std::string, Tensor> tensor_map;
    tensor_map.reserve(model.weights.size() + inputs.size() + 32);
    for (const auto& [k, v] : model.weights) tensor_map[k] = v;
    for (const auto& [k, v] : inputs)         tensor_map[k] = v;

    // 3. Topological order. Throws on cycle.
    const auto sorted = topological_sort(model.graph.node);

    // 4. Walk the sorted nodes, dispatching each one.
    for (const auto& node : sorted) {
        const auto& op = node.op_type;
        const auto& ins  = node.inputs;
        const auto& outs = node.outputs;

        // --- MatMul --------------------------------------------------
        if (op == "MatMul") {
            if (ins.size() < 2) throw std::runtime_error(
                "MatMul '" + node.name + "': need 2 inputs, got " +
                std::to_string(ins.size()));
            const Tensor& A = require_tensor(tensor_map, ins[0]);
            const Tensor& B = require_tensor(tensor_map, ins[1]);
            Tensor Y = ops::matmul(A, B);
            for (const auto& o : outs) tensor_map[o] = Y;
        }
        // --- Gemm ----------------------------------------------------
        else if (op == "Gemm") {
            if (ins.size() < 2) throw std::runtime_error(
                "Gemm '" + node.name + "': need >=2 inputs, got " +
                std::to_string(ins.size()));
            const Tensor& A = require_tensor(tensor_map, ins[0]);
            const Tensor& B = require_tensor(tensor_map, ins[1]);
            Tensor C;
            if (ins.size() >= 3 && !ins[2].empty()) {
                C = require_tensor(tensor_map, ins[2]);
            }
            const float alpha = attr_float(flatten_attrs(node.attributes),
                                           "alpha", 1.0f);
            const float beta  = attr_float(flatten_attrs(node.attributes),
                                           "beta",  1.0f);
            const int transA = attr_int(flatten_attrs(node.attributes),
                                        "transA", 0);
            const int transB = attr_int(flatten_attrs(node.attributes),
                                        "transB", 0);
            Tensor Y = ops::gemm(A, B, C, alpha, beta, transA, transB);
            for (const auto& o : outs) tensor_map[o] = Y;
        }
        // --- Activations (all take a single tensor + an attrs map) ---
        else if (op == "Relu" || op == "Sigmoid" ||
                 op == "Softmax" || op == "Gelu" ||
                 op == "Tanh" || op == "LeakyRelu" || op == "Elu") {
            if (ins.empty()) throw std::runtime_error(
                op + " '" + node.name + "': need 1 input, got 0");
            const Tensor& X = require_tensor(tensor_map, ins[0]);
            const auto flat_attrs = flatten_attrs(node.attributes);
            Tensor Y = [&]{
                if (op == "Relu")     return ops::relu_forward(X, flat_attrs);
                if (op == "Sigmoid")  return ops::sigmoid_forward(X, flat_attrs);
                if (op == "Softmax")  return ops::softmax_forward(X, flat_attrs);
                if (op == "Gelu")     return ops::gelu_forward(X, flat_attrs);
                // Tanh / LeakyRelu / Elu aren't in the dispatch table
                // Issue #9 promises — but accepting them with a clear
                // "not implemented" is friendlier than throwing on an
                // unknown op_type.
                throw std::runtime_error(
                    "run_inference: operator '" + op +
                    "' is on the dispatch list but not yet implemented");
            }();
            for (const auto& o : outs) tensor_map[o] = Y;
        }
        // --- Conv ----------------------------------------------------
        else if (op == "Conv") {
            if (ins.size() < 2) throw std::runtime_error(
                "Conv '" + node.name + "': need >=2 inputs, got " +
                std::to_string(ins.size()));
            const Tensor& X = require_tensor(tensor_map, ins[0]);
            const Tensor& W = require_tensor(tensor_map, ins[1]);
            // Optional bias. An empty string in `ins[2]` (ONNX allows
            // it when no bias initializer is given) means skip bias.
            Tensor B;  // rank-0 = skip
            if (ins.size() >= 3 && !ins[2].empty()) {
                B = require_tensor(tensor_map, ins[2]);
            }
            const auto fa = flatten_attrs(node.attributes);
            ops::ConvParams p;
            // ONNX stores strides/pads/kernel_shape as IntArray. The
            // flat_attrs fallback only sees Int (single int) — so for
            // these we prefer reading from node.attributes directly.
            auto find_int_array = [&](const std::string& k,
                                      std::vector<int64_t>& out) {
                auto it = node.attributes.find(k);
                if (it == node.attributes.end()) return false;
                if (it->second.type == Attribute::Type::IntArray) {
                    out = it->second.ints;
                    return true;
                }
                if (it->second.type == Attribute::Type::Int) {
                    out = {it->second.i};
                    return true;
                }
                return false;
            };
            std::vector<int64_t> strides = {1, 1};
            std::vector<int64_t> pads    = {0, 0, 0, 0};
            find_int_array("strides", strides);
            find_int_array("pads",    pads);
            if (strides.size() >= 1) p.stride_h = static_cast<int>(strides[0]);
            if (strides.size() >= 2) p.stride_w = static_cast<int>(strides[1]);
            if (pads.size() >= 1)    p.pad_h    = static_cast<int>(pads[0]);
            if (pads.size() >= 2)    p.pad_w    = static_cast<int>(pads[1]);
            p.groups = attr_int(flatten_attrs(node.attributes), "group", 1);
            // Dilations default to 1; conv2d doesn't support them in
            // Issue #7, so we silently ignore the attribute.
            Tensor Y = ops::conv2d_forward(X, W, B, p);
            for (const auto& o : outs) tensor_map[o] = Y;
        }
        // --- MaxPool / AveragePool ----------------------------------
        else if (op == "MaxPool" || op == "AveragePool") {
            if (ins.empty()) throw std::runtime_error(
                op + " '" + node.name + "': need 1 input, got 0");
            const Tensor& X = require_tensor(tensor_map, ins[0]);

            // Same IntArray problem as Conv. ONNX MaxPool requires
            // kernel_shape (must be present); strides/pads default
            // to 1 / 0. We read directly from the Attribute union.
            auto find_int_array = [&](const std::string& k,
                                      std::vector<int64_t>& out,
                                      std::vector<int64_t> dflt) {
                auto it = node.attributes.find(k);
                if (it == node.attributes.end()) { out = dflt; return; }
                if (it->second.type == Attribute::Type::IntArray) {
                    out = it->second.ints;
                } else if (it->second.type == Attribute::Type::Int) {
                    out = {it->second.i};
                } else {
                    out = dflt;
                }
            };
            std::vector<int64_t> kernel, strides, pads;
            find_int_array("kernel_shape", kernel, {1, 1});
            find_int_array("strides",      strides, {1, 1});
            find_int_array("pads",         pads, {0, 0, 0, 0});
            if (kernel.size() < 2) throw std::runtime_error(
                op + ": kernel_shape must have >=2 ints");
            const int kH = static_cast<int>(kernel[0]);
            const int kW = static_cast<int>(kernel[1]);
            const int sH = static_cast<int>(strides[0]);
            const int sW = static_cast<int>(strides.size() >= 2 ? strides[1] : strides[0]);
            const int pH = static_cast<int>(pads[0]);
            const int pW = static_cast<int>(pads.size() >= 2 ? pads[1] : pads[0]);

            Tensor Y = (op == "MaxPool")
                ? ops::maxpool_forward(X, kH, kW, sH, sW, pH, pW)
                : ops::avgpool_forward(X, kH, kW, sH, sW, pH, pW);
            for (const auto& o : outs) tensor_map[o] = Y;
        }
        // --- GlobalAveragePool --------------------------------------
        // Issue #10 — added so the executor can run MobileNetV2 and
        // ResNet18 end-to-end. The op has no attributes in practice
        // (no kernel, no stride, no pad: the window IS the full
        // spatial plane).
        else if (op == "GlobalAveragePool") {
            if (ins.empty()) throw std::runtime_error(
                "GlobalAveragePool '" + node.name + "': need 1 input");
            const Tensor& X = require_tensor(tensor_map, ins[0]);
            Tensor Y = ops::global_avgpool_forward(X);
            for (const auto& o : outs) tensor_map[o] = Y;
        }
        // --- BatchNormalization -------------------------------------
        else if (op == "BatchNormalization") {
            // ONNX v15 BatchNormalization: X, scale, B, input_mean,
            // input_var. Crucible implements inference mode only.
            if (ins.size() < 5) throw std::runtime_error(
                "BatchNormalization '" + node.name +
                "': need 5 inputs (X, scale, B, mean, var), got " +
                std::to_string(ins.size()));
            const Tensor& X  = require_tensor(tensor_map, ins[0]);
            const Tensor& S  = require_tensor(tensor_map, ins[1]);
            const Tensor& B  = require_tensor(tensor_map, ins[2]);
            const Tensor& Mu = require_tensor(tensor_map, ins[3]);
            const Tensor& Vr = require_tensor(tensor_map, ins[4]);
            const float eps = attr_float(flatten_attrs(node.attributes),
                                         "epsilon", 1e-5f);
            Tensor Y = ops::batchnorm_forward(X, S, B, Mu, Vr, eps);
            for (const auto& o : outs) tensor_map[o] = Y;
        }
        // --- Flatten -------------------------------------------------
        else if (op == "Flatten") {
            // Flatten has a single data input and an optional `axis`
            // attribute. axis=1 means "keep axis 0, flatten the rest"
            // (the canonical PyTorch nn.Flatten() behaviour).
            if (ins.empty()) throw std::runtime_error(
                "Flatten '" + node.name + "': need 1 input");
            const Tensor& X = require_tensor(tensor_map, ins[0]);
            const int axis = attr_int(flatten_attrs(node.attributes),
                                      "axis", 1);
            if (axis <= 0) {
                // axis=0 or negative flattens everything to a 1-D
                // vector. Our `flatten()` does exactly that.
                Tensor Y = X.flatten();
                for (const auto& o : outs) tensor_map[o] = Y;
            } else {
                // axis>0: keep dims [0..axis), flatten the rest.
                // We materialise the new shape, then reshape.
                int64_t keep = 1;
                for (int i = 0; i < axis && i < static_cast<int>(X.shape().size()); ++i) {
                    if (keep > INT64_MAX / X.shape()[i]) {
                        throw std::overflow_error(
                            "Flatten: leading dimension product overflows int64");
                    }
                    keep *= X.shape()[i];
                }
                std::vector<int64_t> new_shape;
                new_shape.reserve(2);
                new_shape.push_back(keep);
                new_shape.push_back(X.size() / keep);
                Tensor Y = X.reshape(new_shape);
                for (const auto& o : outs) tensor_map[o] = Y;
            }
        }
        // --- Reshape -------------------------------------------------
        else if (op == "Reshape") {
            // ONNX Reshape: input[0]=data, input[1]=shape (1-D int64
            // tensor). The shape tensor lives in int_initializers
            // (when constant) or in tensor_map (when produced by a
            // prior Shape/Slice node — not yet supported here).
            if (ins.size() < 2) throw std::runtime_error(
                "Reshape '" + node.name + "': need 2 inputs (data, shape)");
            const Tensor& X = require_tensor(tensor_map, ins[0]);
            std::vector<int64_t> new_shape;
            auto it_int = model.int_initializers.find(ins[1]);
            if (it_int != model.int_initializers.end()) {
                new_shape = it_int->second;
            } else {
                const Tensor& shape_t = require_tensor(tensor_map, ins[1]);
                new_shape.reserve(shape_t.size());
                for (int64_t i = 0; i < shape_t.size(); ++i) {
                    new_shape.push_back(static_cast<int64_t>(shape_t.data()[i]));
                }
            }

            // Resolve 0 and -1 dims per ONNX spec
            int negative_one_idx = -1;
            int64_t other_prod = 1;
            for (size_t i = 0; i < new_shape.size(); ++i) {
                if (new_shape[i] == -1) {
                    if (negative_one_idx != -1) {
                        throw std::runtime_error("Reshape: multiple -1 dimensions in target shape");
                    }
                    negative_one_idx = static_cast<int>(i);
                } else if (new_shape[i] == 0) {
                    if (i < X.shape().size()) {
                        new_shape[i] = X.shape()[i];
                    }
                    other_prod *= new_shape[i];
                } else {
                    other_prod *= new_shape[i];
                }
            }
            if (negative_one_idx != -1) {
                if (other_prod == 0) {
                    throw std::runtime_error("Reshape: division by zero when resolving -1 dimension");
                }
                new_shape[negative_one_idx] = X.size() / other_prod;
            }

            Tensor Y = X.reshape(new_shape);
            for (const auto& o : outs) tensor_map[o] = Y;
        }
        // --- Add -----------------------------------------------------
        else if (op == "Add") {
            // Element-wise / channel-broadcast add. See `add_broadcast`.
            if (ins.size() < 2) throw std::runtime_error(
                "Add '" + node.name + "': need 2 inputs, got " +
                std::to_string(ins.size()));
            const Tensor& A = require_tensor(tensor_map, ins[0]);
            const Tensor& B = require_tensor(tensor_map, ins[1]);
            Tensor Y = add_broadcast(A, B);
            for (const auto& o : outs) tensor_map[o] = Y;
        }
        // --- Concat --------------------------------------------------
        else if (op == "Concat") {
            // Concat along axis (1 by default for channel-concat in
            // NCHW). Implementation is axis-0 only — see concat_axis0.
            // Issue #11's MobileNetV2 wiring has no concat nodes, but
            // some day we will.
            if (ins.empty()) throw std::runtime_error(
                "Concat '" + node.name + "': need >=1 input");
            std::vector<Tensor> parts;
            parts.reserve(ins.size());
            for (const auto& n : ins) parts.push_back(require_tensor(tensor_map, n));
            Tensor Y = concat_axis0(parts);
            for (const auto& o : outs) tensor_map[o] = Y;
        }
        // --- Identity / Dropout (passthrough in inference) ----------
        else if (op == "Identity" || op == "Dropout") {
            if (ins.empty()) throw std::runtime_error(
                op + " '" + node.name + "': need 1 input");
            const Tensor& X = require_tensor(tensor_map, ins[0]);
            Tensor Y = X;  // share the underlying buffer — passthrough
            for (const auto& o : outs) tensor_map[o] = Y;
        }
        // --- Known-but-unimplemented ops with explicit messages -------
        else if (op == "Clip" || op == "Shape" || op == "Gather" ||
                 op == "GatherElements" || op == "Unsqueeze" ||
                 op == "Squeeze" || op == "Resize") {
            throw std::runtime_error(
                "run_inference: op '" + op + "' on node '" + node.name +
                "' is recognized but not yet supported by Crucible");
        }
        // --- Unknown -------------------------------------------------
        else {
            throw std::runtime_error(
                "run_inference: unknown op_type '" + op +
                "' on node '" + node.name + "'");
        }
    }

    // 5. Return the tensor bound to the model's first listed output.
    //    ONNX graphs can have multiple outputs; Issue #9's plan calls
    //    for returning the first one. A future issue will wrap the
    //    result in a Session that returns the full vector.
    if (model.output_names.empty()) {
        throw std::runtime_error(
            "run_inference: model has no output_names");
    }
    return require_tensor(tensor_map, model.output_names[0]);
}

}  // namespace crucible