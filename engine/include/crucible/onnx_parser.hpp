// ONNX model parser.
//
// Parses .onnx files into crucible::Model with a hand-rolled, hermetic
// protobuf wire-format decoder (no libprotobuf, no protoc step). It covers
// the ONNX subset needed for inference:
//
//   * ModelProto     -> graph, opset_import (default-domain version)
//   * GraphProto     -> node[], initializer[], input[], output[]
//   * NodeProto      -> op_type, domain, name, inputs[], outputs[], attribute[]
//   * AttributeProto -> float / int / string / floats / ints
//   * TensorProto    -> dims, data_type, raw_data / float_data / int64_data;
//                       external-data tensors are rejected
//   * ValueInfoProto -> name, tensor elem_type, shape (symbolic dims -> -1)
//
// Wire format: https://protobuf.dev/programming-guides/encoding/. Each
// field is a `(field_number << 3 | wire_type)` varint followed by its
// payload. Wire types 0 (VARINT), 1 (FIXED64), 2 (LEN) and 5 (FIXED32) are
// decoded or skipped; anything else, and any truncated payload, throws.

#pragma once

#include "crucible/tensor.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace crucible {

// A single attribute on a node. MobileNetV2 uses only float and int;
// we expose those as a tagged union.
struct Attribute {
    enum class Type { Float, Int, FloatArray, IntArray, String };
    Type type = Type::Float;
    float f = 0.0f;
    int64_t i = 0;
    std::string s;
    std::vector<float> floats;
    std::vector<int64_t> ints;
};

// A single node in the computation graph.
struct GraphNode {
    std::string op_type;                                      // "MatMul", "Relu", ...
    std::string name;                                         // optional user-provided name
    std::string domain;                                       // "" or "ai.onnx" = default domain
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::unordered_map<std::string, Attribute> attributes;    // by attribute name
};

// Declared type of a graph input/output (ValueInfoProto). `shape` uses -1
// for symbolic or missing dims; `has_shape` is false when the model gives
// no shape at all. `elem_type` is the ONNX TensorProto.DataType code
// (1 = FLOAT, 7 = INT64, ...; 0 = unknown).
struct ValueInfo {
    std::string name;
    int32_t elem_type = 0;
    bool has_shape = false;
    std::vector<int64_t> shape;
};

// Top-level graph: nodes + initializers + IO lists.
struct Graph {
    std::string name;
    std::vector<GraphNode> node;
    // initializer payloads stored as Tensors (float32) or raw int64
    // buffers, depending on data_type. We expose:
    //   - weights:        float initializers as Tensor
    //   - int_initializers: non-float initializers as a vector of int64
    std::unordered_map<std::string, Tensor> weights;
    std::unordered_map<std::string, std::vector<int64_t>> int_initializers;
    // Initializers whose dtype Crucible cannot represent (name -> ONNX
    // data_type code). Kept so the executor can say *why* a tensor is
    // missing instead of reporting a dangling edge.
    std::unordered_map<std::string, int32_t> unsupported_initializers;
    std::vector<std::string> input_names;    // runtime inputs (initializers excluded)
    std::vector<std::string> output_names;
    std::vector<ValueInfo> inputs;           // same order as input_names (may be empty
    std::vector<ValueInfo> outputs;          // for hand-built graphs in unit tests)
    // Default-domain ("" / "ai.onnx") opset version from ModelProto.opset_import.
    // 0 = not declared (hand-built graphs); treated as the latest semantics.
    int64_t opset = 0;
};

// Top-level Model struct (per ENGINEERING_PLAN.md §4).
struct Model {
    Graph graph;
    std::unordered_map<std::string, Tensor>& weights = graph.weights;
    std::unordered_map<std::string, std::vector<int64_t>>& int_initializers =
        graph.int_initializers;
    std::vector<std::string>& input_names  = graph.input_names;
    std::vector<std::string>& output_names = graph.output_names;

    Model() = default;
    Model(const Model& other)
        : graph(other.graph) {}
    Model(Model&& other) noexcept
        : graph(std::move(other.graph)) {}
    Model& operator=(const Model& other) {
        if (this != &other) {
            graph = other.graph;
        }
        return *this;
    }
    Model& operator=(Model&& other) noexcept {
        if (this != &other) {
            graph = std::move(other.graph);
        }
        return *this;
    }
};

// Load a .onnx file from disk and parse it into a Model.
// Throws std::runtime_error on I/O failure or malformed input.
Model load_model(const std::string& path);

}  // namespace crucible
