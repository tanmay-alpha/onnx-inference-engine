// Graph executor — see executor.hpp for the public API.
//
// Two pieces:
//
//   1. Ordering — Kahn's BFS over the node list, treating "input name = X"
//      and "some node outputs X" as an edge. `topological_sort` returns a
//      reordered copy (used by tests); the executor uses the index form to
//      avoid copying every node on every inference.
//
//   2. run_inference — walks the ordered nodes and dispatches each one to
//      its operator. Tensors are resolved without copying: initializers and
//      caller inputs are read in place, only node outputs are owned by the
//      run, and each intermediate is freed right after its last consumer
//      runs, so peak memory is bounded by the live activations, not the
//      sum of every activation in the graph.
//
// Unsupported features are rejected with a message naming the node, the
// operator and the attribute. `check_model` runs the same per-node checks
// statically so callers (the API server) can reject a model at upload
// time without executing it.

#include "crucible/executor.hpp"

#include "crucible/ops/activations.hpp"
#include "crucible/ops/conv2d.hpp"
#include "crucible/ops/linear.hpp"
#include "crucible/ops/norm.hpp"
#include "crucible/ops/pooling.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace crucible {

namespace {

constexpr int32_t kOnnxFloat = 1;

// --------------------------------------------------------------------
// Ordering
// --------------------------------------------------------------------

std::vector<std::size_t> topological_order(const std::vector<GraphNode>& nodes) {
    // name -> producing node. If two nodes claim the same output (malformed
    // model) the first one wins; duplicate names are easier to debug than
    // a crash.
    std::unordered_map<std::string, std::size_t> producer;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        for (const auto& out_name : nodes[i].outputs) {
            if (out_name.empty()) continue;
            producer.emplace(out_name, i);
        }
    }

    std::vector<int> in_degree(nodes.size(), 0);
    std::vector<std::vector<std::size_t>> consumers(nodes.size());
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        for (const auto& in_name : nodes[i].inputs) {
            auto it = producer.find(in_name);
            if (it == producer.end()) continue;   // initializer / graph input
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

    std::queue<std::size_t> ready;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        if (in_degree[i] == 0) ready.push(i);
    }

    std::vector<std::size_t> order;
    order.reserve(nodes.size());
    while (!ready.empty()) {
        const std::size_t i = ready.front();
        ready.pop();
        order.push_back(i);
        for (std::size_t c : consumers[i]) {
            if (--in_degree[c] == 0) ready.push(c);
        }
    }

    if (order.size() != nodes.size()) {
        throw std::runtime_error(
            "topological_sort: cycle detected (sorted " +
            std::to_string(order.size()) + " of " +
            std::to_string(nodes.size()) + " nodes)");
    }
    return order;
}

// --------------------------------------------------------------------
// Small utilities
// --------------------------------------------------------------------

std::string shape_str(const std::vector<int64_t>& s) {
    std::string out = "[";
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (i) out += ", ";
        out += std::to_string(s[i]);
    }
    return out + "]";
}

std::string node_label(const GraphNode& n) {
    return n.op_type + " node '" + n.name + "'";
}

[[noreturn]] void node_error(const GraphNode& n, const std::string& msg) {
    throw std::runtime_error(node_label(n) + ": " + msg);
}

const char* onnx_dtype_name(int32_t t) {
    switch (t) {
        case 1: return "float32";   case 2: return "uint8";
        case 3: return "int8";      case 4: return "uint16";
        case 5: return "int16";     case 6: return "int32";
        case 7: return "int64";     case 8: return "string";
        case 9: return "bool";      case 10: return "float16";
        case 11: return "float64";  case 12: return "uint32";
        case 13: return "uint64";   case 16: return "bfloat16";
        default: return "unknown";
    }
}

// --------------------------------------------------------------------
// Attribute accessors (read the parsed AttributeProto union directly)
// --------------------------------------------------------------------

const Attribute* find_attr(const GraphNode& n, const char* key) {
    auto it = n.attributes.find(key);
    return it == n.attributes.end() ? nullptr : &it->second;
}

int64_t attr_int(const GraphNode& n, const char* key, int64_t dflt) {
    const Attribute* a = find_attr(n, key);
    if (!a) return dflt;
    if (a->type == Attribute::Type::Int) return a->i;
    node_error(n, std::string("attribute '") + key + "' must be an int");
}

float attr_float(const GraphNode& n, const char* key, float dflt) {
    const Attribute* a = find_attr(n, key);
    if (!a) return dflt;
    float v;
    if (a->type == Attribute::Type::Float) v = a->f;
    else if (a->type == Attribute::Type::Int) v = static_cast<float>(a->i);
    else node_error(n, std::string("attribute '") + key + "' must be a float");
    if (!std::isfinite(v)) {
        node_error(n, std::string("attribute '") + key + "' is not finite");
    }
    return v;
}

std::vector<int64_t> attr_ints(const GraphNode& n, const char* key,
                               std::vector<int64_t> dflt) {
    const Attribute* a = find_attr(n, key);
    if (!a) return dflt;
    if (a->type == Attribute::Type::IntArray) return a->ints;
    if (a->type == Attribute::Type::Int) return {a->i};
    node_error(n, std::string("attribute '") + key + "' must be a list of ints");
}

std::string attr_string(const GraphNode& n, const char* key, std::string dflt) {
    const Attribute* a = find_attr(n, key);
    if (!a) return dflt;
    if (a->type == Attribute::Type::String) return a->s;
    node_error(n, std::string("attribute '") + key + "' must be a string");
}

int checked_int(const GraphNode& n, int64_t v, const char* what) {
    if (v < INT_MIN || v > INT_MAX) {
        node_error(n, std::string(what) + " value " + std::to_string(v) +
                          " is out of range");
    }
    return static_cast<int>(v);
}

// --------------------------------------------------------------------
// Static support checks (shared by check_model and the dispatcher)
// --------------------------------------------------------------------

const std::vector<std::string>& supported_op_list() {
    static const std::vector<std::string> ops = {
        "Add", "AveragePool", "BatchNormalization", "Concat", "Conv",
        "Dropout", "Elu", "Flatten", "Gelu", "Gemm", "GlobalAveragePool",
        "Identity", "LeakyRelu", "MatMul", "MaxPool", "Relu", "Reshape",
        "Sigmoid", "Softmax", "Tanh",
    };
    return ops;
}

bool is_default_domain(const std::string& d) {
    return d.empty() || d == "ai.onnx";
}

// 2-D sliding-window attributes shared by Conv / MaxPool / AveragePool.
struct Window2D {
    int kH = 0, kW = 0;   // 0 = take from weight (Conv)
    int sH = 1, sW = 1;
    int pH = 0, pW = 0;
};

Window2D window_attrs(const GraphNode& n, bool kernel_required) {
    const std::string auto_pad = attr_string(n, "auto_pad", "NOTSET");
    if (auto_pad != "NOTSET" && auto_pad != "VALID") {
        node_error(n, "auto_pad='" + auto_pad +
                          "' is not supported (only NOTSET or VALID)");
    }
    for (int64_t d : attr_ints(n, "dilations", {})) {
        if (d != 1) node_error(n, "dilations other than 1 are not supported");
    }

    Window2D w;
    const auto kernel = attr_ints(n, "kernel_shape", {});
    if (kernel_required && kernel.empty()) {
        node_error(n, "missing required attribute 'kernel_shape'");
    }
    if (!kernel.empty()) {
        if (kernel.size() != 2) {
            node_error(n, "only 2-D windows are supported (kernel_shape has " +
                              std::to_string(kernel.size()) + " dims)");
        }
        w.kH = checked_int(n, kernel[0], "kernel_shape");
        w.kW = checked_int(n, kernel[1], "kernel_shape");
    }

    const auto strides = attr_ints(n, "strides", {1, 1});
    if (strides.size() != 2) {
        node_error(n, "strides must have 2 values for a 2-D window");
    }
    w.sH = checked_int(n, strides[0], "strides");
    w.sW = checked_int(n, strides[1], "strides");

    if (auto_pad == "NOTSET") {
        const auto pads = attr_ints(n, "pads", {0, 0, 0, 0});
        if (pads.size() != 4) {
            node_error(n, "pads must have 4 values [top, left, bottom, right]");
        }
        if (pads[0] != pads[2] || pads[1] != pads[3]) {
            node_error(n, "asymmetric pads " + shape_str(pads) +
                              " are not supported (begin must equal end)");
        }
        w.pH = checked_int(n, pads[0], "pads");
        w.pW = checked_int(n, pads[1], "pads");
    }
    return w;
}

bool has_used_output(const GraphNode& n, std::size_t index) {
    return n.outputs.size() > index && !n.outputs[index].empty();
}

// Throws std::runtime_error describing the first reason `n` cannot run.
void check_node(const GraphNode& n, int64_t opset) {
    if (!is_default_domain(n.domain)) {
        node_error(n, "operator domain '" + n.domain +
                          "' is not supported (only the default ONNX domain)");
    }
    const auto& ops = supported_op_list();
    if (std::find(ops.begin(), ops.end(), n.op_type) == ops.end()) {
        throw std::runtime_error(
            "run_inference: unsupported op_type '" + n.op_type +
            "' on node '" + n.name + "'");
    }
    const std::string& op = n.op_type;
    if (op == "Conv") {
        (void)window_attrs(n, false);
        if (attr_int(n, "group", 1) < 1) node_error(n, "group must be >= 1");
    } else if (op == "MaxPool" || op == "AveragePool") {
        (void)window_attrs(n, true);
        if (attr_int(n, "ceil_mode", 0) != 0) {
            node_error(n, "ceil_mode=1 is not supported");
        }
        if (op == "MaxPool") {
            if (attr_int(n, "storage_order", 0) != 0) {
                node_error(n, "storage_order=1 is not supported");
            }
            if (has_used_output(n, 1)) {
                node_error(n, "the optional Indices output is not supported");
            }
        }
    } else if (op == "BatchNormalization") {
        if (attr_int(n, "spatial", 1) != 1) {
            node_error(n, "spatial=0 is not supported");
        }
        if (attr_int(n, "training_mode", 0) != 0) {
            node_error(n, "training_mode=1 is not supported (inference only)");
        }
        for (std::size_t i = 1; i < n.outputs.size(); ++i) {
            if (!n.outputs[i].empty()) {
                node_error(n, "training-mode outputs (running mean/var) are not supported");
            }
        }
    } else if (op == "Gelu") {
        const std::string a = attr_string(n, "approximate", "none");
        if (a != "none" && a != "tanh") {
            node_error(n, "approximate='" + a + "' is not a valid value");
        }
    } else if (op == "Reshape") {
        if (attr_int(n, "allowzero", 0) != 0) {
            node_error(n, "allowzero=1 is not supported");
        }
    } else if (op == "Concat") {
        if (!find_attr(n, "axis")) node_error(n, "missing required attribute 'axis'");
    } else if (op == "Dropout") {
        if (has_used_output(n, 1)) {
            node_error(n, "the optional mask output is not supported");
        }
    } else if (op == "Softmax") {
        (void)opset;  // any axis is supported; semantics depend on opset
    }
}

// --------------------------------------------------------------------
// Tensor kernels used only by the executor
// --------------------------------------------------------------------

// Multidirectional (numpy-style) broadcasting Add.
Tensor broadcast_add(const Tensor& A, const Tensor& B) {
    if (A.size() == 0 || B.size() == 0) {
        throw std::invalid_argument(
            "Add: empty operand (shapes " + shape_str(A.shape()) + " and " +
            shape_str(B.shape()) + ")");
    }
    const auto& as = A.shape();
    const auto& bs = B.shape();

    if (as == bs) {
        Tensor out(as, 0.0f);
        if (as.empty()) return Tensor({}, std::vector<float>{A.data()[0] + B.data()[0]});
        const float* a = A.data();
        const float* b = B.data();
        float* y = out.data();
        for (int64_t i = 0; i < out.size(); ++i) y[i] = a[i] + b[i];
        return out;
    }

    const std::size_t rank = std::max(as.size(), bs.size());
    std::vector<int64_t> out_shape(rank), a_dims(rank, 1), b_dims(rank, 1);
    for (std::size_t i = 0; i < as.size(); ++i) a_dims[rank - as.size() + i] = as[i];
    for (std::size_t i = 0; i < bs.size(); ++i) b_dims[rank - bs.size() + i] = bs[i];
    for (std::size_t i = 0; i < rank; ++i) {
        if (a_dims[i] != b_dims[i] && a_dims[i] != 1 && b_dims[i] != 1) {
            throw std::invalid_argument(
                "Add: shapes " + shape_str(as) + " and " + shape_str(bs) +
                " are not broadcast-compatible");
        }
        out_shape[i] = std::max(a_dims[i], b_dims[i]);
    }
    if (rank == 0) {
        return Tensor({}, std::vector<float>{A.data()[0] + B.data()[0]});
    }

    // Element strides into A/B; 0 along broadcast (size-1) axes.
    std::vector<int64_t> a_st(rank, 0), b_st(rank, 0);
    int64_t sa = 1, sb = 1;
    for (std::size_t i = rank; i-- > 0;) {
        a_st[i] = a_dims[i] == 1 ? 0 : sa;
        b_st[i] = b_dims[i] == 1 ? 0 : sb;
        sa *= a_dims[i];
        sb *= b_dims[i];
    }

    Tensor out(out_shape, 0.0f);
    const float* a = A.data();
    const float* b = B.data();
    float* y = out.data();
    const int64_t total = out.size();
    const int64_t inner = out_shape.back();
    const int64_t a_in = a_st.back();
    const int64_t b_in = b_st.back();
    std::vector<int64_t> idx(rank, 0);
    int64_t a_off = 0, b_off = 0;
    for (int64_t o = 0; o < total; o += inner) {
        for (int64_t k = 0; k < inner; ++k) {
            y[o + k] = a[a_off + k * a_in] + b[b_off + k * b_in];
        }
        // Odometer increment over the outer axes.
        for (std::size_t d = rank - 1; d-- > 0;) {
            ++idx[d];
            a_off += a_st[d];
            b_off += b_st[d];
            if (idx[d] < out_shape[d]) break;
            a_off -= a_st[d] * out_shape[d];
            b_off -= b_st[d] * out_shape[d];
            idx[d] = 0;
        }
    }
    return out;
}

Tensor concat(const std::vector<const Tensor*>& parts, int64_t axis_attr) {
    const auto& ref = parts[0]->shape();
    const int64_t rank = static_cast<int64_t>(ref.size());
    if (rank == 0) throw std::invalid_argument("Concat: inputs must have rank >= 1");
    int64_t axis = axis_attr < 0 ? axis_attr + rank : axis_attr;
    if (axis < 0 || axis >= rank) {
        throw std::invalid_argument("Concat: axis " + std::to_string(axis_attr) +
                                    " out of range for rank " + std::to_string(rank));
    }
    std::vector<int64_t> out_shape = ref;
    out_shape[axis] = 0;
    for (const Tensor* p : parts) {
        const auto& s = p->shape();
        if (static_cast<int64_t>(s.size()) != rank) {
            throw std::invalid_argument("Concat: all inputs must have the same rank");
        }
        for (int64_t d = 0; d < rank; ++d) {
            if (d != axis && s[d] != ref[d]) {
                throw std::invalid_argument(
                    "Concat: shapes " + shape_str(ref) + " and " + shape_str(s) +
                    " differ outside axis " + std::to_string(axis));
            }
        }
        out_shape[axis] += s[axis];
    }
    int64_t outer = 1;
    for (int64_t d = 0; d < axis; ++d) outer *= ref[d];
    int64_t trailing = 1;
    for (int64_t d = axis + 1; d < rank; ++d) trailing *= ref[d];

    Tensor out(out_shape, 0.0f);
    const int64_t out_chunk = out_shape[axis] * trailing;
    int64_t offset = 0;
    for (const Tensor* p : parts) {
        const int64_t chunk = p->shape()[axis] * trailing;
        for (int64_t o = 0; o < outer; ++o) {
            std::copy(p->data() + o * chunk, p->data() + (o + 1) * chunk,
                      out.data() + o * out_chunk + offset);
        }
        offset += chunk;
    }
    return out;
}

// --------------------------------------------------------------------
// Execution environment: zero-copy lookup + ownership of intermediates
// --------------------------------------------------------------------

class TensorEnv {
public:
    TensorEnv(const Model& model,
              const std::unordered_map<std::string, Tensor>& inputs)
        : model_(model), inputs_(inputs) {}

    const Tensor& get(const GraphNode& node, const std::string& name) const {
        if (auto it = owned_.find(name); it != owned_.end()) return it->second;
        if (auto it = inputs_.find(name); it != inputs_.end()) return it->second;
        if (auto it = model_.weights.find(name); it != model_.weights.end()) return it->second;
        if (auto it = model_.graph.unsupported_initializers.find(name);
            it != model_.graph.unsupported_initializers.end()) {
            node_error(node, "input '" + name + "' is a " +
                                 onnx_dtype_name(it->second) +
                                 " initializer; Crucible tensors are float32 "
                                 "(int64 only for shape inputs)");
        }
        throw std::runtime_error(
            "run_inference: tensor '" + name + "' (input of " +
            node_label(node) + ") is neither a graph input, an initializer, "
            "nor a prior node's output");
    }

    const Tensor& input(const GraphNode& node, std::size_t i) const {
        if (node.inputs.size() <= i || node.inputs[i].empty()) {
            node_error(node, "missing required input #" + std::to_string(i));
        }
        return get(node, node.inputs[i]);
    }

    bool has_input(const GraphNode& node, std::size_t i) const {
        return node.inputs.size() > i && !node.inputs[i].empty();
    }

    void put(const GraphNode& node, Tensor t) {
        if (node.outputs.empty() || node.outputs[0].empty()) {
            node_error(node, "node has no output");
        }
        auto& slot = owned_[node.outputs[0]];
        live_ -= slot.size();
        live_ += t.size();
        if (max_live_ > 0 && live_ > max_live_) {
            throw ResourceLimitError(
                "intermediate tensors need " + std::to_string(live_) +
                " elements in total after " + node_label(node) +
                ", above the configured limit of " + std::to_string(max_live_));
        }
        slot = std::move(t);
    }

    void release(const std::string& name) {
        if (auto it = owned_.find(name); it != owned_.end()) {
            live_ -= it->second.size();
            owned_.erase(it);
        }
    }

    void set_live_limit(int64_t max_live) { max_live_ = max_live; }

    Tensor take(const std::string& name) {
        if (auto it = owned_.find(name); it != owned_.end()) {
            Tensor t = std::move(it->second);
            owned_.erase(it);
            return t;
        }
        // Graph output that is directly an input or initializer.
        if (auto it = inputs_.find(name); it != inputs_.end()) return it->second;
        if (auto it = model_.weights.find(name); it != model_.weights.end()) return it->second;
        throw std::runtime_error(
            "run_inference: graph output '" + name + "' was never produced");
    }

private:
    const Model& model_;
    const std::unordered_map<std::string, Tensor>& inputs_;
    std::unordered_map<std::string, Tensor> owned_;
    int64_t live_ = 0;      // elements held in owned_
    int64_t max_live_ = 0;  // 0 = unlimited
};

void validate_inputs(const Model& model,
                     const std::unordered_map<std::string, Tensor>& inputs) {
    for (std::size_t i = 0; i < model.input_names.size(); ++i) {
        const std::string& name = model.input_names[i];
        if (model.weights.count(name) || model.int_initializers.count(name)) continue;
        auto it = inputs.find(name);
        if (it == inputs.end()) {
            throw std::invalid_argument(
                "run_inference: no input provided for graph input '" + name + "'");
        }
        if (i >= model.graph.inputs.size()) continue;   // hand-built graph
        const ValueInfo& vi = model.graph.inputs[i];
        if (vi.elem_type != 0 && vi.elem_type != kOnnxFloat) {
            throw std::invalid_argument(
                "run_inference: graph input '" + name + "' is declared as " +
                onnx_dtype_name(vi.elem_type) +
                "; Crucible only accepts float32 graph inputs");
        }
        if (!vi.has_shape) continue;
        const auto& got = it->second.shape();
        bool ok = got.size() == vi.shape.size();
        for (std::size_t d = 0; ok && d < got.size(); ++d) {
            if (vi.shape[d] > 0 && vi.shape[d] != got[d]) ok = false;
        }
        if (!ok) {
            throw std::invalid_argument(
                "run_inference: graph input '" + name + "' expects shape " +
                shape_str(vi.shape) + " (-1 = any size), got " + shape_str(got));
        }
    }
}

// --------------------------------------------------------------------
// Per-operator dispatch
// --------------------------------------------------------------------

Tensor run_node(const GraphNode& node, const Model& model, const TensorEnv& env) {
    const std::string& op = node.op_type;
    const int64_t opset = model.graph.opset;

    if (op == "MatMul") {
        return ops::matmul(env.input(node, 0), env.input(node, 1));
    }
    if (op == "Gemm") {
        const Tensor& A = env.input(node, 0);
        const Tensor& B = env.input(node, 1);
        const bool transA = attr_int(node, "transA", 0) != 0;
        const bool transB = attr_int(node, "transB", 0) != 0;
        const float alpha = attr_float(node, "alpha", 1.0f);
        const float beta = attr_float(node, "beta", 1.0f);
        Tensor C;
        if (env.has_input(node, 2)) {
            const Tensor& Cin = env.input(node, 2);
            if (A.rank() != 2 || B.rank() != 2) {
                node_error(node, "A and B must be rank 2");
            }
            const int64_t M = transA ? A.shape()[1] : A.shape()[0];
            const int64_t N = transB ? B.shape()[0] : B.shape()[1];
            const std::vector<int64_t> mn{M, N};
            if (Cin.shape() == mn) {
                C = Cin;
            } else {
                // Unidirectional broadcast of C to (M, N).
                C = broadcast_add(Tensor(mn, 0.0f), Cin);
                if (C.shape() != mn) {
                    node_error(node, "C of shape " + shape_str(Cin.shape()) +
                                         " cannot be broadcast to " + shape_str(mn));
                }
            }
        }
        return ops::gemm(A, B, C, alpha, beta, transA, transB);
    }
    if (op == "Relu" || op == "Sigmoid" || op == "Tanh") {
        static const std::unordered_map<std::string, float> none;
        const Tensor& X = env.input(node, 0);
        if (op == "Relu") return ops::relu_forward(X, none);
        if (op == "Sigmoid") return ops::sigmoid_forward(X, none);
        return ops::tanh_forward(X, none);
    }
    if (op == "LeakyRelu") {
        return ops::leaky_relu_forward(env.input(node, 0),
                                       {{"alpha", attr_float(node, "alpha", 0.01f)}});
    }
    if (op == "Elu") {
        return ops::elu_forward(env.input(node, 0),
                                {{"alpha", attr_float(node, "alpha", 1.0f)}});
    }
    if (op == "Gelu") {
        const bool tanh_approx = attr_string(node, "approximate", "none") == "tanh";
        return ops::gelu_forward(env.input(node, 0),
                                 {{"approximate", tanh_approx ? 1.0f : 0.0f}});
    }
    if (op == "Softmax") {
        const Tensor& X = env.input(node, 0);
        const int64_t rank = X.rank();
        if (opset != 0 && opset < 13) {
            // Opset 1-12: coerce to 2-D [prod(shape[:axis]), prod(shape[axis:])]
            // and normalise each row; default axis = 1.
            int64_t axis = attr_int(node, "axis", 1);
            if (axis < 0) axis += rank;
            if (axis < 0 || axis > rank) {
                node_error(node, "axis out of range for rank " + std::to_string(rank));
            }
            int64_t outer = 1;
            for (int64_t d = 0; d < axis; ++d) outer *= X.shape()[d];
            const int64_t inner = outer == 0 ? 0 : X.size() / outer;
            Tensor flat = X.reshape({outer, inner});
            Tensor y = ops::softmax_forward(flat, {{"axis", -1.0f}});
            return y.reshape(X.shape());
        }
        return ops::softmax_forward(
            X, {{"axis", static_cast<float>(attr_int(node, "axis", -1))}});
    }
    if (op == "Conv") {
        const Tensor& X = env.input(node, 0);
        const Tensor& W = env.input(node, 1);
        Tensor bias;
        if (env.has_input(node, 2)) bias = env.input(node, 2);
        const Window2D w = window_attrs(node, false);
        if (w.kH != 0 && W.rank() == 4 &&
            (w.kH != W.shape()[2] || w.kW != W.shape()[3])) {
            node_error(node, "kernel_shape does not match weight shape " +
                                 shape_str(W.shape()));
        }
        ops::ConvParams p;
        p.stride_h = w.sH;
        p.stride_w = w.sW;
        p.pad_h = w.pH;
        p.pad_w = w.pW;
        p.groups = checked_int(node, attr_int(node, "group", 1), "group");
        return ops::conv2d_forward(X, W, bias, p);
    }
    if (op == "MaxPool" || op == "AveragePool") {
        const Tensor& X = env.input(node, 0);
        const Window2D w = window_attrs(node, true);
        if (op == "MaxPool") {
            return ops::maxpool_forward(X, w.kH, w.kW, w.sH, w.sW, w.pH, w.pW);
        }
        const bool include_pad = attr_int(node, "count_include_pad", 0) != 0;
        return ops::avgpool_forward(X, w.kH, w.kW, w.sH, w.sW, w.pH, w.pW, include_pad);
    }
    if (op == "GlobalAveragePool") {
        return ops::global_avgpool_forward(env.input(node, 0));
    }
    if (op == "BatchNormalization") {
        return ops::batchnorm_forward(env.input(node, 0), env.input(node, 1),
                                      env.input(node, 2), env.input(node, 3),
                                      env.input(node, 4),
                                      attr_float(node, "epsilon", 1e-5f));
    }
    if (op == "Flatten") {
        const Tensor& X = env.input(node, 0);
        const int64_t rank = X.rank();
        int64_t axis = attr_int(node, "axis", 1);
        if (axis < 0) axis += rank;
        if (axis < 0 || axis > rank) {
            node_error(node, "axis out of range for rank " + std::to_string(rank));
        }
        int64_t outer = 1;
        for (int64_t d = 0; d < axis; ++d) outer *= X.shape()[d];
        int64_t inner = 1;
        for (int64_t d = axis; d < rank; ++d) inner *= X.shape()[d];
        return X.reshape({outer, inner});
    }
    if (op == "Reshape") {
        const Tensor& X = env.input(node, 0);
        if (!env.has_input(node, 1)) node_error(node, "missing shape input");
        std::vector<int64_t> shape;
        auto it_int = model.int_initializers.find(node.inputs[1]);
        if (it_int != model.int_initializers.end()) {
            shape = it_int->second;
        } else {
            const Tensor& s = env.input(node, 1);
            shape.reserve(static_cast<std::size_t>(s.size()));
            for (int64_t i = 0; i < s.size(); ++i) {
                const float v = s.data()[i];
                // Out-of-range or fractional float -> int64 conversion is UB.
                if (!std::isfinite(v) || v != std::trunc(v) || std::fabs(v) > 9.0e15f) {
                    node_error(node, "shape input must contain integers, got " +
                                         std::to_string(v));
                }
                shape.push_back(static_cast<int64_t>(v));
            }
        }
        int neg_idx = -1;
        int64_t known = 1;
        for (std::size_t i = 0; i < shape.size(); ++i) {
            if (shape[i] == -1) {
                if (neg_idx != -1) node_error(node, "more than one -1 in target shape");
                neg_idx = static_cast<int>(i);
                continue;
            }
            if (shape[i] == 0) {
                if (i >= X.shape().size()) {
                    node_error(node, "0 in target shape at an axis the input lacks");
                }
                shape[i] = X.shape()[i];
            } else if (shape[i] < 0) {
                node_error(node, "invalid dimension " + std::to_string(shape[i]));
            }
            known *= shape[i];
        }
        if (neg_idx != -1) {
            if (known == 0 || X.size() % known != 0) {
                node_error(node, "cannot infer -1 dimension reshaping " +
                                     shape_str(X.shape()) + " to " + shape_str(shape));
            }
            shape[neg_idx] = X.size() / known;
        }
        return X.reshape(shape);
    }
    if (op == "Add") {
        return broadcast_add(env.input(node, 0), env.input(node, 1));
    }
    if (op == "Concat") {
        std::vector<const Tensor*> parts;
        for (std::size_t i = 0; i < node.inputs.size(); ++i) {
            parts.push_back(&env.input(node, i));
        }
        if (parts.empty()) node_error(node, "needs at least one input");
        return concat(parts, attr_int(node, "axis", 0));
    }
    if (op == "Identity" || op == "Dropout") {
        return env.input(node, 0);
    }
    throw std::runtime_error("run_inference: no kernel for op_type '" + op + "'");
}

}  // namespace

// ====================================================================
// Public API
// ====================================================================

std::vector<GraphNode> topological_sort(const std::vector<GraphNode>& nodes) {
    std::vector<GraphNode> sorted;
    sorted.reserve(nodes.size());
    for (std::size_t i : topological_order(nodes)) sorted.push_back(nodes[i]);
    return sorted;
}

const std::vector<std::string>& supported_ops() { return supported_op_list(); }

std::vector<std::string> check_model(const Model& model) {
    std::vector<std::string> problems;
    for (std::size_t i = 0; i < model.graph.inputs.size(); ++i) {
        const ValueInfo& vi = model.graph.inputs[i];
        if (vi.elem_type != 0 && vi.elem_type != kOnnxFloat) {
            problems.push_back("graph input '" + vi.name + "' is " +
                               onnx_dtype_name(vi.elem_type) +
                               "; only float32 graph inputs are supported");
        }
    }
    for (const auto& node : model.graph.node) {
        try {
            check_node(node, model.graph.opset);
        } catch (const std::exception& e) {
            problems.emplace_back(e.what());
        }
        for (const auto& in : node.inputs) {
            auto it = model.graph.unsupported_initializers.find(in);
            if (it != model.graph.unsupported_initializers.end()) {
                problems.push_back(node_label(node) + ": input '" + in + "' is a " +
                                   onnx_dtype_name(it->second) + " initializer");
            }
        }
    }
    try {
        (void)topological_order(model.graph.node);
    } catch (const std::exception& e) {
        problems.emplace_back(e.what());
    }
    if (model.output_names.empty()) problems.emplace_back("model declares no outputs");
    return problems;
}

Tensor run_inference(const Model& model,
                     const std::unordered_map<std::string, Tensor>& inputs,
                     const RunOptions& options) {
    ScopedElementLimit limit(options.max_tensor_elements);
    ScopedDeadline deadline(options.timeout_ms);
    validate_inputs(model, inputs);
    if (model.output_names.empty()) {
        throw std::runtime_error("run_inference: model has no output_names");
    }

    const auto& nodes = model.graph.node;
    const auto order = topological_order(nodes);

    // Last consumer (position in `order`) of every tensor name, so each
    // intermediate can be freed as soon as nothing else reads it. Only the
    // first graph output is returned, so it is the only one kept to the end;
    // outputs nobody consumes are freed immediately.
    std::unordered_map<std::string, std::size_t> last_use;
    for (std::size_t pos = 0; pos < order.size(); ++pos) {
        for (const auto& in : nodes[order[pos]].inputs) last_use[in] = pos;
    }
    const std::string& result_name = model.output_names[0];

    TensorEnv env(model, inputs);
    env.set_live_limit(options.max_total_elements);
    for (std::size_t pos = 0; pos < order.size(); ++pos) {
        check_deadline("between nodes");
        const GraphNode& node = nodes[order[pos]];
        check_node(node, model.graph.opset);
        env.put(node, run_node(node, model, env));
        const std::string& out = node.outputs[0];
        if (out != result_name && !last_use.count(out)) env.release(out);
        for (const auto& in : node.inputs) {
            auto it = last_use.find(in);
            if (it != last_use.end() && it->second == pos && in != result_name) {
                env.release(in);
            }
        }
    }
    return env.take(model.output_names[0]);
}

}  // namespace crucible
