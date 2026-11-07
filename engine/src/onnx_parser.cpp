// ONNX parser implementation — see onnx_parser.hpp for the design notes.
//
// The file is organized in three layers:
//
//   1. Wire-format reader (Cursor): reads varints, tags, sub-message
//      boundaries from a contiguous byte buffer. Throws on truncated
//      input. This is the only layer that knows about protobuf.
//
//   2. ONNX message parsers (parse_tensor, parse_node, parse_graph,
//      parse_model): each consumes a sub-message and returns the typed
//      domain object. Each field is read directly; unknown fields are
//      skipped via skip_field() rather than raising an error so we
//      stay forward-compatible.
//
//   3. Public API: load_model(path) — read the file, dispatch to
//      parse_model, return.
//
// All wire types follow the protobuf spec; see
// https://protobuf.dev/programming-guides/encoding/.

#include "crucible/onnx_parser.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <limits>
#include <vector>

namespace crucible {

// ONNX TensorProto data_type values (from onnx.proto TensorProto).
// We only care about FLOAT(1) and INT64(7) for Issue #4.
namespace onnx_dtype {
constexpr int32_t FLOAT  = 1;
constexpr int32_t INT64  = 7;
}  // namespace onnx_dtype

// Helper: compute product of int64 dims; returns 1 for empty dims
// (matches protobuf empty-list semantics for scalar shapes). Declared
// at file scope so the message parsers below can use it.
static int64_t product_i64(const std::vector<int64_t>& dims) {
    int64_t p = 1;
    for (auto d : dims) {
        if (d < 0) {
            throw std::runtime_error("ONNX: negative dimension in initializer");
        }
        if (d > 0 && p > std::numeric_limits<int64_t>::max() / d) {
            throw std::runtime_error("ONNX: initializer shape product overflows int64_t");
        }
        p *= d;
    }
    return p;
}

// ----------------------------------------------------------------------
// Layer 1: Wire-format reader
// ----------------------------------------------------------------------

// Non-owning view of bytes inside the model buffer.
struct ByteView {
    const std::uint8_t* ptr = nullptr;
    std::size_t len = 0;
    const std::uint8_t* data() const { return ptr; }
    std::size_t size() const { return len; }
    bool empty() const { return len == 0; }
};

class Cursor {
public:
    explicit Cursor(const std::uint8_t* data, std::size_t size)
        : data_(data), size_(size) {}

    bool eof() const { return pos_ >= size_; }
    std::size_t pos() const { return pos_; }
    std::size_t remaining() const { return size_ - pos_; }

    // Read a base-128 varint. The MSB set on each byte signals
    // continuation. Throws if the varint is malformed (overflow or
    // longer than 10 bytes — the protobuf limit for 64-bit values).
    std::uint64_t read_varint() {
        std::uint64_t result = 0;
        int shift = 0;
        for (int i = 0; i < 10; ++i) {
            if (pos_ >= size_) {
                throw std::runtime_error("ONNX: truncated varint");
            }
            std::uint8_t b = data_[pos_++];
            result |= static_cast<std::uint64_t>(b & 0x7f) << shift;
            if ((b & 0x80) == 0) {
                return result;
            }
            shift += 7;
        }
        throw std::runtime_error("ONNX: varint exceeds 10 bytes");
    }

    // Field tag = (field_number << 3) | wire_type.
    struct Tag {
        std::uint32_t field;
        std::uint8_t  wire;
    };

    Tag read_tag() {
        std::size_t tag_pos = pos_;
        std::uint64_t v = read_varint();
        std::uint32_t field = static_cast<std::uint32_t>(v >> 3);
        std::uint8_t  wire  = static_cast<std::uint8_t>(v & 0x7);
        if (field == 0) {
            throw std::runtime_error("ONNX: tag has field number 0 at pos " + std::to_string(tag_pos));
        }
        return {field, wire};
    }

    // Read a length-delimited payload (string / bytes / sub-message). The
    // result is a view into the caller's buffer, which outlives the parse;
    // nothing is copied, so peak memory stays ~1x the file size.
    ByteView read_length_delimited() {
        std::uint64_t len = read_varint();
        if (len > remaining()) {
            throw std::runtime_error("ONNX: length-delimited payload exceeds buffer");
        }
        ByteView out{data_ + pos_, static_cast<std::size_t>(len)};
        pos_ += static_cast<std::size_t>(len);
        return out;
    }

    // Read a 4-byte little-endian value (FIXED32).
    std::uint32_t read_fixed32() {
        if (remaining() < 4) {
            throw std::runtime_error("ONNX: truncated fixed32");
        }
        std::uint32_t v;
        std::memcpy(&v, data_ + pos_, 4);
        pos_ += 4;
        return v;
    }

    // Read an 8-byte little-endian value (FIXED64).
    std::uint64_t read_fixed64() {
        if (remaining() < 8) {
            throw std::runtime_error("ONNX: truncated fixed64");
        }
        std::uint64_t v;
        std::memcpy(&v, data_ + pos_, 8);
        pos_ += 8;
        return v;
    }

    // Skip a field of the given wire type. Used for unknown fields so
    // the parser is forward-compatible with newer ONNX specs.
    void skip_field(std::uint8_t wire) {
        switch (wire) {
            case 0: (void)read_varint();                  break;
            case 1: (void)read_fixed64();                break;
            case 2: { std::uint64_t n = read_varint();
                      if (n > remaining()) {
                          throw std::runtime_error("ONNX: skip length exceeds buffer");
                      }
                      pos_ += static_cast<std::size_t>(n); break; }
            case 5: (void)read_fixed32();                break;
            default:
                throw std::runtime_error(
                    "ONNX: unknown wire type " + std::to_string(wire));
        }
    }

private:
    const std::uint8_t* data_;
    std::size_t size_;
    std::size_t pos_ = 0;
};

// Iterate over (field, wire) pairs in a sub-message, calling fn for each.
// Unknown fields are skipped automatically.
template <typename Fn>
void for_each_field(Cursor& c, Fn&& fn) {
    while (!c.eof()) {
        auto tag = c.read_tag();
        fn(tag.field, tag.wire);
        // The handler must consume the payload — except for fields it
        // doesn't recognise, in which case the caller calls
        // c.skip_field(wire).
        //
        // We can't auto-skip here because we don't know if the handler
        // consumed the payload or not. Instead, callers call
        // skip_unknown(c, field, wire) inside their default branch.
        (void)tag;  // suppress -Wunused-result
    }
}

// ----------------------------------------------------------------------
// Layer 2: ONNX message parsers
// ----------------------------------------------------------------------

// Decode a 32-bit IEEE-754 float from a little-endian uint32.
static float bits_to_float(std::uint32_t bits) {
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// Parse a TensorProto. Returns either a Tensor (for FLOAT) or a
// std::vector<int64_t> (for INT64). The caller selects the appropriate
// container via the `dtype` field.
//
// We support:
//   - field 1: dims (packed repeated int64) — [1, 1] for scalars
//   - field 2: data_type (int32)
//   - field 8: name (string)
//   - field 9: raw_data (bytes) — packed per data_type
//   - field 13: float_data (packed repeated float) — convenience for FLOAT
//
// Other fields (string_data, int32_data, etc.) are skipped; if we ever
// hit them in a real model we'll add them.
struct TensorParseResult {
    std::string name;
    int32_t data_type = -1;
    std::vector<int64_t> dims;
    std::vector<float>    float_data;
    std::vector<int64_t>  int64_data;
    ByteView raw_data;
    bool external = false;   // data_location == EXTERNAL (weights in a side file)
};

static TensorParseResult parse_tensor(Cursor& c) {
    TensorParseResult t;
    while (!c.eof()) {
        auto tag = c.read_tag();
        switch (tag.field) {
            case 1: {  // dims (repeated int64)
                if (tag.wire == 2) {
                    auto bytes = c.read_length_delimited();
                    Cursor inner(bytes.data(), bytes.size());
                    while (!inner.eof()) {
                        t.dims.push_back(static_cast<int64_t>(inner.read_varint()));
                    }
                } else {
                    t.dims.push_back(static_cast<int64_t>(c.read_varint()));
                }
                break;
            }
            case 2:  // data_type (int32)
                t.data_type = static_cast<int32_t>(c.read_varint());
                break;
            case 4: {  // float_data (repeated float)
                if (tag.wire == 2) {
                    auto bytes = c.read_length_delimited();
                    Cursor inner(bytes.data(), bytes.size());
                    while (!inner.eof()) {
                        t.float_data.push_back(bits_to_float(inner.read_fixed32()));
                    }
                } else {
                    t.float_data.push_back(bits_to_float(c.read_fixed32()));
                }
                break;
            }
            case 7: {  // int64_data (repeated int64)
                if (tag.wire == 2) {
                    auto bytes = c.read_length_delimited();
                    Cursor inner(bytes.data(), bytes.size());
                    while (!inner.eof()) {
                        t.int64_data.push_back(static_cast<int64_t>(inner.read_varint()));
                    }
                } else {
                    t.int64_data.push_back(static_cast<int64_t>(c.read_varint()));
                }
                break;
            }
            case 8: {  // name
                auto bytes = c.read_length_delimited();
                t.name.assign(reinterpret_cast<const char*>(bytes.data()),
                              bytes.size());
                break;
            }
            case 9: {  // raw_data (bytes)
                t.raw_data = c.read_length_delimited();
                break;
            }
            case 14:  // data_location (enum: 0 DEFAULT, 1 EXTERNAL)
                t.external = c.read_varint() == 1;
                break;
            default:
                c.skip_field(tag.wire);
                break;
        }
    }

    if (t.external) {
        throw std::runtime_error(
            "ONNX: initializer '" + t.name + "' uses external data "
            "(weights stored outside the .onnx file); Crucible only loads "
            "self-contained models");
    }

    if (!t.raw_data.empty()) {
        if (t.data_type == onnx_dtype::FLOAT) {
            if (t.raw_data.size() % sizeof(float) != 0) {
                throw std::runtime_error(
                    "ONNX: float raw_data size not multiple of 4");
            }
            t.float_data.resize(t.raw_data.size() / sizeof(float));
            std::memcpy(t.float_data.data(), t.raw_data.data(),
                        t.raw_data.size());
        } else if (t.data_type == onnx_dtype::INT64) {
            if (t.raw_data.size() % sizeof(int64_t) != 0) {
                throw std::runtime_error(
                    "ONNX: int64 raw_data size not multiple of 8");
            }
            t.int64_data.resize(t.raw_data.size() / sizeof(int64_t));
            std::memcpy(t.int64_data.data(), t.raw_data.data(),
                        t.raw_data.size());
        }
    }

    return t;
}

// Parse an AttributeProto. We materialise it into an Attribute struct.
//   field 1 = name (string)
//   field 2 = f   (float,  via fixed32)
//   field 3 = i   (int64,  via varint)
//   field 4 = s   (bytes,  via length-delimited)
//   field 7 = ints  (packed repeated int64)
//   field 8 = floats (packed repeated float)
static Attribute parse_attribute(Cursor& c) {
    Attribute a;
    while (!c.eof()) {
        auto tag = c.read_tag();
        switch (tag.field) {
            case 1: {  // name
                auto bytes = c.read_length_delimited();
                a.s.assign(reinterpret_cast<const char*>(bytes.data()),
                           bytes.size());
                // Attribute uses `s` field as the NAME when name is set
                // via a separate field. Since we store the name in the
                // map key, we don't need the value here; just consume.
                break;
            }
            case 2:  // f (float) — fixed32
                a.type = Attribute::Type::Float;
                a.f = bits_to_float(c.read_fixed32());
                break;
            case 3:  // i (int64) — varint
                a.type = Attribute::Type::Int;
                a.i = static_cast<int64_t>(c.read_varint());
                break;
            case 4: {  // s (bytes) — used for string-valued attributes
                auto bytes = c.read_length_delimited();
                a.type = Attribute::Type::String;
                a.s.assign(reinterpret_cast<const char*>(bytes.data()),
                           bytes.size());
                break;
            }
            case 7: {  // floats (repeated float)
                if (tag.wire == 2) {
                    auto bytes = c.read_length_delimited();
                    Cursor inner(bytes.data(), bytes.size());
                    while (!inner.eof()) {
                        a.floats.push_back(bits_to_float(inner.read_fixed32()));
                    }
                } else {
                    a.floats.push_back(bits_to_float(c.read_fixed32()));
                }
                a.type = Attribute::Type::FloatArray;
                break;
            }
            case 8: {  // ints (repeated int64)
                if (tag.wire == 2) {
                    auto bytes = c.read_length_delimited();
                    Cursor inner(bytes.data(), bytes.size());
                    while (!inner.eof()) {
                        a.ints.push_back(static_cast<int64_t>(inner.read_varint()));
                    }
                } else {
                    a.ints.push_back(static_cast<int64_t>(c.read_varint()));
                }
                a.type = Attribute::Type::IntArray;
                break;
            }
            default:
                c.skip_field(tag.wire);
                break;
        }
    }
    // Extract name from a.s if it wasn't set — but actually, the name is
    // a separate field; we set it in the calling parser, not here.
    return a;
}

// Parse a NodeProto. Returns the GraphNode ready to insert into the
// graph. The parse fills in op_type, name, inputs, outputs, and
// attributes; the public GraphNode struct is reused for the parsed
// result since it has the right shape.
static GraphNode parse_node(Cursor& c) {
    GraphNode n;
    while (!c.eof()) {
        auto tag = c.read_tag();
        switch (tag.field) {
            case 1: {  // input (string, repeated)
                auto bytes = c.read_length_delimited();
                n.inputs.emplace_back(
                    reinterpret_cast<const char*>(bytes.data()), bytes.size());
                break;
            }
            case 2: {  // output (string, repeated)
                auto bytes = c.read_length_delimited();
                n.outputs.emplace_back(
                    reinterpret_cast<const char*>(bytes.data()), bytes.size());
                break;
            }
            case 3: {  // name (string)
                auto bytes = c.read_length_delimited();
                n.name.assign(reinterpret_cast<const char*>(bytes.data()),
                              bytes.size());
                break;
            }
            case 7: {  // domain (string)
                auto bytes = c.read_length_delimited();
                n.domain.assign(reinterpret_cast<const char*>(bytes.data()),
                                bytes.size());
                break;
            }
            case 4: {  // op_type (string)
                auto bytes = c.read_length_delimited();
                n.op_type.assign(reinterpret_cast<const char*>(bytes.data()),
                                 bytes.size());
                break;
            }
            case 5: {  // attribute (AttributeProto, repeated)
                auto bytes = c.read_length_delimited();
                // parse_attribute populates the value; the name is in
                // field 1 of the sub-message — re-scan to find it.
                Cursor inner(bytes.data(), bytes.size());
                Attribute attr = parse_attribute(inner);
                std::string attr_name;
                Cursor inner2(bytes.data(), bytes.size());
                while (!inner2.eof()) {
                    auto t = inner2.read_tag();
                    if (t.field == 1) {
                        auto nb = inner2.read_length_delimited();
                        attr_name.assign(reinterpret_cast<const char*>(nb.data()),
                                         nb.size());
                        break;
                    } else {
                        inner2.skip_field(t.wire);
                    }
                }
                n.attributes[attr_name] = std::move(attr);
                break;
            }
            default:
                c.skip_field(tag.wire);
                break;
        }
    }
    return n;
}

// Parse a ValueInfoProto: name (1) and type (2) -> TypeProto.tensor_type (1)
// -> { elem_type (1), shape (2) -> TensorShapeProto.dim (1, repeated) ->
// { dim_value (1) | dim_param (2) } }.
static std::string read_string(Cursor& c) {
    auto bytes = c.read_length_delimited();
    return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

static void parse_shape(Cursor& c, ValueInfo& v) {
    v.has_shape = true;
    while (!c.eof()) {
        auto tag = c.read_tag();
        if (tag.field == 1 && tag.wire == 2) {  // dim
            auto bytes = c.read_length_delimited();
            Cursor dim(bytes.data(), bytes.size());
            int64_t value = -1;  // symbolic / unknown
            while (!dim.eof()) {
                auto dt = dim.read_tag();
                if (dt.field == 1 && dt.wire == 0) {
                    value = static_cast<int64_t>(dim.read_varint());
                } else {
                    dim.skip_field(dt.wire);
                }
            }
            v.shape.push_back(value);
        } else {
            c.skip_field(tag.wire);
        }
    }
}

static ValueInfo parse_value_info(Cursor& c) {
    ValueInfo v;
    while (!c.eof()) {
        auto tag = c.read_tag();
        if (tag.field == 1 && tag.wire == 2) {
            v.name = read_string(c);
        } else if (tag.field == 2 && tag.wire == 2) {  // TypeProto
            auto type_bytes = c.read_length_delimited();
            Cursor type(type_bytes.data(), type_bytes.size());
            while (!type.eof()) {
                auto tt = type.read_tag();
                if (tt.field == 1 && tt.wire == 2) {  // tensor_type
                    auto tensor_bytes = type.read_length_delimited();
                    Cursor tensor(tensor_bytes.data(), tensor_bytes.size());
                    while (!tensor.eof()) {
                        auto et = tensor.read_tag();
                        if (et.field == 1 && et.wire == 0) {
                            v.elem_type = static_cast<int32_t>(tensor.read_varint());
                        } else if (et.field == 2 && et.wire == 2) {
                            auto shape_bytes = tensor.read_length_delimited();
                            Cursor shape(shape_bytes.data(), shape_bytes.size());
                            parse_shape(shape, v);
                        } else {
                            tensor.skip_field(et.wire);
                        }
                    }
                } else {
                    type.skip_field(tt.wire);
                }
            }
        } else {
            c.skip_field(tag.wire);
        }
    }
    return v;
}

// Parse a GraphProto.
static Graph parse_graph(Cursor& c) {
    Graph g;
    while (!c.eof()) {
        auto tag = c.read_tag();
        switch (tag.field) {
            case 1: {  // node (NodeProto, repeated)
                auto bytes = c.read_length_delimited();
                Cursor inner(bytes.data(), bytes.size());
                g.node.push_back(parse_node(inner));
                break;
            }
            case 2: {  // name (string)
                auto bytes = c.read_length_delimited();
                g.name.assign(reinterpret_cast<const char*>(bytes.data()),
                              bytes.size());
                break;
            }
            case 5: {  // initializer (TensorProto, repeated)
                auto bytes = c.read_length_delimited();
                Cursor inner(bytes.data(), bytes.size());
                TensorParseResult tp = parse_tensor(inner);
                if (tp.name.empty()) {
                    throw std::runtime_error(
                        "ONNX: initializer with empty name");
                }
                if (tp.data_type == onnx_dtype::FLOAT) {
                    // Construct Tensor from dims + float_data.
                    int64_t expected_size = tp.dims.empty() ? 1LL : product_i64(tp.dims);
                    if ((int64_t)tp.float_data.size() != expected_size) {
                        throw std::runtime_error(
                            "ONNX: float initializer size mismatch for '" + tp.name +
                            "' (dims=" + std::to_string(tp.dims.size()) +
                            ", expected=" + std::to_string(expected_size) +
                            ", float_data=" + std::to_string(tp.float_data.size()) +
                            ", raw_data=" + std::to_string(tp.raw_data.size()) + ")");
                    }
                    g.weights[tp.name] = Tensor(std::move(tp.dims), std::move(tp.float_data));
                } else if (tp.data_type == onnx_dtype::INT64) {
                    int64_t expected_size = tp.dims.empty() ? 1LL : product_i64(tp.dims);
                    if ((int64_t)tp.int64_data.size() != expected_size) {
                        throw std::runtime_error(
                            "ONNX: int64 initializer size mismatch for '" + tp.name +
                            "' (dims=" + std::to_string(tp.dims.size()) +
                            ", expected=" + std::to_string(expected_size) +
                            ", int64_data=" + std::to_string(tp.int64_data.size()) +
                            ", raw_data=" + std::to_string(tp.raw_data.size()) + ")");
                    }
                    g.int_initializers[tp.name] = std::move(tp.int64_data);
                } else {
                    // No Tensor representation (int32, float16, bool, ...).
                    // Remember it so a consuming node fails with a precise
                    // message rather than "dangling edge".
                    g.unsupported_initializers[tp.name] = tp.data_type;
                }
                break;
            }
            case 11: {  // input (ValueInfoProto, repeated)
                auto bytes = c.read_length_delimited();
                Cursor inner(bytes.data(), bytes.size());
                g.inputs.push_back(parse_value_info(inner));
                break;
            }
            case 12: {  // output (ValueInfoProto, repeated)
                auto bytes = c.read_length_delimited();
                Cursor inner(bytes.data(), bytes.size());
                g.outputs.push_back(parse_value_info(inner));
                break;
            }
            default:
                c.skip_field(tag.wire);
                break;
        }
    }
    // IR < 4 models list every initializer in graph.input as well; those are
    // not runtime inputs, so drop them.
    std::vector<ValueInfo> runtime_inputs;
    runtime_inputs.reserve(g.inputs.size());
    for (auto& vi : g.inputs) {
        if (g.weights.count(vi.name) || g.int_initializers.count(vi.name) ||
            g.unsupported_initializers.count(vi.name)) {
            continue;
        }
        g.input_names.push_back(vi.name);
        runtime_inputs.push_back(std::move(vi));
    }
    g.inputs = std::move(runtime_inputs);
    for (const auto& vi : g.outputs) g.output_names.push_back(vi.name);
    return g;
}

// Parse a ModelProto: graph (7) and opset_import (8). Only the default
// domain's opset version is kept — it selects operator semantics that
// changed across opsets (e.g. Softmax before/after opset 13).
static Model parse_model(Cursor& c) {
    Model m;
    int64_t opset = 0;
    while (!c.eof()) {
        auto tag = c.read_tag();
        if (tag.field == 7 && tag.wire == 2) {
            auto bytes = c.read_length_delimited();
            Cursor inner(bytes.data(), bytes.size());
            m.graph = parse_graph(inner);
        } else if (tag.field == 8 && tag.wire == 2) {
            auto bytes = c.read_length_delimited();
            Cursor inner(bytes.data(), bytes.size());
            std::string domain;
            int64_t version = 0;
            while (!inner.eof()) {
                auto t = inner.read_tag();
                if (t.field == 1 && t.wire == 2) {
                    domain = read_string(inner);
                } else if (t.field == 2 && t.wire == 0) {
                    version = static_cast<int64_t>(inner.read_varint());
                } else {
                    inner.skip_field(t.wire);
                }
            }
            if (domain.empty() || domain == "ai.onnx") opset = version;
        } else {
            c.skip_field(tag.wire);
        }
    }
    m.graph.opset = opset;
    return m;
}

// ----------------------------------------------------------------------
// Layer 3: Public API
// ----------------------------------------------------------------------

Model load_model(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("ONNX: cannot open file " + path);
    }
    in.seekg(0, std::ios::end);
    std::streamsize size = in.tellg();
    in.seekg(0, std::ios::beg);
    if (size < 0) {
        throw std::runtime_error("ONNX: cannot determine file size of " + path);
    }
    std::vector<std::uint8_t> buf(static_cast<std::size_t>(size));
    if (size > 0 && !in.read(reinterpret_cast<char*>(buf.data()), size)) {
        throw std::runtime_error("ONNX: read error on " + path);
    }
    if (buf.empty()) {
        throw std::runtime_error("ONNX: file " + path + " is empty");
    }
    Cursor c(buf.data(), buf.size());
    return parse_model(c);
}

}  // namespace crucible