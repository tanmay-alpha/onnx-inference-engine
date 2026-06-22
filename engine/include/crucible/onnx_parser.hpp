// ONNX model parser — Issue #4.
//
// Parses .onnx files into crucible::Model (see model.hpp).
//
// Implementation note: this module uses a hand-rolled, hermetic
// protobuf wire-format decoder rather than linking against libprotobuf.
// The submodule `engine/third_party/protobuf` would let us use the
// generated C++ stubs from onnx.proto, but that requires the submodule
// to be initialized AND a `protoc` build step. The hand-rolled reader
// keeps the build hermetic (no submodules, no protoc, no link deps) and
// covers exactly the ONNX subset we need for inference:
//
//   * ModelProto  -> GraphProto
//   * GraphProto  -> node[], initializer[], input[], output[]
//   * NodeProto   -> op_type, name, inputs[], outputs[], attribute[]
//   * AttributeProto -> name -> float / int (the only types MobileNetV2 uses)
//   * TensorProto -> dims, data_type, raw_data / float_data
//
// The wire format spec is in
// https://protobuf.dev/programming-guides/encoding/. A field is encoded
// as: `(field_number << 3 | wire_type)` varint, then payload.
//
// Wire types we handle:
//   0 = VARINT      (int32 / int64 / bool / enum)
//   1 = FIXED64     (double / fixed64 / sfixed64)
//   2 = LEN_DELIM   (string / bytes / sub-message / packed repeated)
//   5 = FIXED32     (float / fixed32 / sfixed32)
//
// All other wire types are skipped (with a clear error if they would
// lose data we care about, e.g. a 64-bit dim we don't expect).
//
// This module is a transitional implementation. Once the protobuf
// submodule is initialized, the same public API (Model + load_model) can
// be re-implemented over onnx::ModelProto without changing the test
// suite or downstream code.

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
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::unordered_map<std::string, Attribute> attributes;    // by attribute name
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
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
};

// Top-level Model struct (per ENGINEERING_PLAN.md §4).
struct Model {
    Graph graph;
    std::unordered_map<std::string, Tensor>& weights = graph.weights;
    std::unordered_map<std::string, std::vector<int64_t>>& int_initializers =
        graph.int_initializers;
    std::vector<std::string>& input_names  = graph.input_names;
    std::vector<std::string>& output_names = graph.output_names;
};

// Load a .onnx file from disk and parse it into a Model.
// Throws std::runtime_error on I/O failure or malformed input.
Model load_model(const std::string& path);

}  // namespace crucible
