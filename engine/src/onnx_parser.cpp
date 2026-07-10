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

    // Read a length-delimited payload (string / bytes / sub-message).
    std::vector<std::uint8_t> read_length_delimited() {
        std::uint64_t len = read_varint();
        if (len > remaining()) {
            throw std::runtime_error("ONNX: length-delimited payload exceeds buffer");
        }
        std::vector<std::uint8_t> out(data_ + pos_, data_ + pos_ + len);
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
    std::vector<std::uint8_t> raw_data;
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
            default:
                c.skip_field(tag.wire);
                break;
        }
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

// Parse a ValueInfoProto. We only need the name (field 1).
struct ValueInfoParseResult {
    std::string name;
};

static ValueInfoParseResult parse_value_info(Cursor& c) {
    ValueInfoParseResult v;
    while (!c.eof()) {
        auto tag = c.read_tag();
        if (tag.field == 1) {
            auto bytes = c.read_length_delimited();
            v.name.assign(reinterpret_cast<const char*>(bytes.data()),
                          bytes.size());
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
                    g.weights[tp.name] = Tensor(tp.dims, tp.float_data);
                } else if (tp.data_type == onnx_dtype::INT64) {
                    g.int_initializers[tp.name] = std::move(tp.int64_data);
                } else {
                    // Unknown dtype — skip silently; we don't have a
                    // representation for it.
                }
                break;
            }
            case 11: {  // input (ValueInfoProto, repeated)
                auto bytes = c.read_length_delimited();
                Cursor inner(bytes.data(), bytes.size());
                g.input_names.push_back(parse_value_info(inner).name);
                break;
            }
            case 12: {  // output (ValueInfoProto, repeated)
                auto bytes = c.read_length_delimited();
                Cursor inner(bytes.data(), bytes.size());
                g.output_names.push_back(parse_value_info(inner).name);
                break;
            }
            default:
                c.skip_field(tag.wire);
                break;
        }
    }
    return g;
}

// Parse a ModelProto. Only field 7 (graph) is required.
static Model parse_model(Cursor& c) {
    Model m;
    while (!c.eof()) {
        auto tag = c.read_tag();
        if (tag.field == 7) {
            auto bytes = c.read_length_delimited();
            Cursor inner(bytes.data(), bytes.size());
            m.graph = parse_graph(inner);
        } else {
            c.skip_field(tag.wire);
        }
    }
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