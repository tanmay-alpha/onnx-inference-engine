//! Crucible WASM runtime: a small, separate pure-Rust ONNX interpreter for the
//! browser. It is NOT the native C++ engine and supports only: Add (numpy
//! multidirectional broadcasting), MatMul (rank-2 operands), Relu, Sigmoid,
//! Softmax (opset-aware: opset >= 13 default axis -1 on that axis only;
//! opset < 13 default axis 1 with 2-D coercion). Models must have exactly one
//! float32 runtime input, float32 initializers, and no external data.
//! Parity vs ONNX Runtime: tests/parity.rs (fixtures from tests/gen_fixtures.py).
//!
//! JS exports (wasm-bindgen):
//! - `runInference(modelBytes: Uint8Array, input: Float32Array, shape: Int32Array): Float32Array`
//! - `runFraudModel(modelBytes, amount, oldBalanceOrig, newBalanceOrig, oldBalanceDest, newBalanceDest, isCashOut, isTransfer): number`
//! - `outputShape(modelBytes: Uint8Array, input: Float32Array, shape: Int32Array): Int32Array`
//!   — runs the model and returns the first output's shape
//! - `supportedOps(): string[]` — `["Add", "MatMul", "Relu", "Sigmoid", "Softmax"]`
//! - `modelInfo(modelBytes: Uint8Array): string` — JSON:
//!   `{opset: number|null, num_nodes, inputs: [{name, elem_type, shape: number[]|null}],
//!   outputs: [...], ops_used: string[], supported: bool, unsupported_reasons: string[]}`
//!   (symbolic dims are -1; `inputs` excludes initializers; non-default-domain
//!   ops are reported as `domain::op`).
//!
//! All functions throw a string error (never return zeros) on unsupported ops,
//! incompatible shapes, or malformed models.

#![allow(clippy::excessive_precision, clippy::too_many_arguments)]

use wasm_bindgen::prelude::*;
use std::collections::HashMap;

/// Largest tensor (in elements) any parsed or computed value may have.
/// 64M f32 = 256 MiB, well above every demo model and far below what would
/// let a hostile model exhaust the browser tab.
pub const MAX_TENSOR_ELEMENTS: usize = 64 * 1024 * 1024;

/// Checked element count of `dims`: rejects negative dims, overflow, and
/// anything above [`MAX_TENSOR_ELEMENTS`].
pub fn checked_numel<I: IntoIterator<Item = i64>>(dims: I, what: &str) -> Result<usize, String> {
    let n = dims.into_iter().try_fold(1usize, |acc, d| {
        let d = usize::try_from(d).map_err(|_| format!("{what}: negative dimension {d}"))?;
        acc.checked_mul(d)
            .filter(|&n| n <= MAX_TENSOR_ELEMENTS)
            .ok_or_else(|| format!("{what}: tensor exceeds {MAX_TENSOR_ELEMENTS} element limit"))
    })?;
    Ok(n)
}

// ============================================================================
// Tensor representation
// ============================================================================

#[derive(Clone, Debug)]
pub struct Tensor {
    pub shape: Vec<i64>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn new(shape: Vec<i64>, data: Vec<f32>) -> Self {
        Tensor { shape, data }
    }

    pub fn empty() -> Self {
        Tensor {
            shape: vec![],
            data: vec![],
        }
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }
}

// ============================================================================
// Protobuf Wire-Format Decoder (Cursor)
// ============================================================================

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Cursor { data, pos: 0 }
    }

    fn eof(&self) -> bool {
        self.pos >= self.data.len()
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn read_varint(&mut self) -> Result<u64, String> {
        let mut result = 0u64;
        let mut shift = 0;
        for _ in 0..10 {
            if self.pos >= self.data.len() {
                return Err("ONNX: truncated varint".into());
            }
            let b = self.data[self.pos];
            self.pos += 1;
            result |= ((b & 0x7f) as u64) << shift;
            if (b & 0x80) == 0 {
                return Ok(result);
            }
            shift += 7;
        }
        Err("ONNX: varint exceeds 10 bytes".into())
    }

    fn read_tag(&mut self) -> Result<(u32, u8), String> {
        let tag = self.read_varint()?;
        let field = (tag >> 3) as u32;
        let wire = (tag & 0x7) as u8;
        if field == 0 {
            return Err("ONNX: tag has field number 0".into());
        }
        Ok((field, wire))
    }

    fn read_length_delimited(&mut self) -> Result<&'a [u8], String> {
        let len = usize::try_from(self.read_varint()?).unwrap_or(usize::MAX);
        if len > self.remaining() {
            return Err("ONNX: length-delimited payload exceeds buffer".into());
        }
        let out = &self.data[self.pos..self.pos + len];
        self.pos += len;
        Ok(out)
    }

    fn read_fixed32(&mut self) -> Result<u32, String> {
        if self.remaining() < 4 {
            return Err("ONNX: truncated fixed32".into());
        }
        let mut bytes = [0u8; 4];
        bytes.copy_from_slice(&self.data[self.pos..self.pos + 4]);
        self.pos += 4;
        Ok(u32::from_le_bytes(bytes))
    }

    fn read_fixed64(&mut self) -> Result<u64, String> {
        if self.remaining() < 8 {
            return Err("ONNX: truncated fixed64".into());
        }
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&self.data[self.pos..self.pos + 8]);
        self.pos += 8;
        Ok(u64::from_le_bytes(bytes))
    }

    fn skip_field(&mut self, wire: u8) -> Result<(), String> {
        match wire {
            0 => {
                self.read_varint()?;
            }
            1 => {
                self.read_fixed64()?;
            }
            2 => {
                let len = usize::try_from(self.read_varint()?).unwrap_or(usize::MAX);
                if len > self.remaining() {
                    return Err("ONNX: skip length exceeds buffer".into());
                }
                self.pos += len;
            }
            5 => {
                self.read_fixed32()?;
            }
            _ => return Err(format!("ONNX: unknown wire type {wire}")),
        }
        Ok(())
    }
}

// ============================================================================
// ONNX Message Parsers
// ============================================================================

#[derive(Clone, Debug, Default)]
pub struct Attribute {
    pub name: String,
    pub f: f32,
    pub i: i64,
    pub s: Vec<u8>,
    pub ints: Vec<i64>,
    pub floats: Vec<f32>,
}

fn parse_attribute(c: &mut Cursor) -> Result<Attribute, String> {
    let mut attr = Attribute::default();
    while !c.eof() {
        let (field, wire) = c.read_tag()?;
        match field {
            1 => {
                attr.name = String::from_utf8(c.read_length_delimited()?.to_vec())
                    .map_err(|_| "ONNX: invalid UTF-8 in attribute name")?;
            }
            2 => {
                let bits = c.read_fixed32()?;
                attr.f = f32::from_bits(bits);
            }
            3 => {
                attr.i = c.read_varint()? as i64;
            }
            4 => {
                attr.s = c.read_length_delimited()?.to_vec();
            }
            // AttributeProto.ints = 8
            8 => {
                if wire == 2 {
                    let bytes = c.read_length_delimited()?;
                    let mut inner = Cursor::new(bytes);
                    while !inner.eof() {
                        attr.ints.push(inner.read_varint()? as i64);
                    }
                } else {
                    attr.ints.push(c.read_varint()? as i64);
                }
            }
            // AttributeProto.floats = 7
            7 => {
                if wire == 2 {
                    let bytes = c.read_length_delimited()?;
                    let mut inner = Cursor::new(bytes);
                    while !inner.eof() {
                        let bits = inner.read_fixed32()?;
                        attr.floats.push(f32::from_bits(bits));
                    }
                } else {
                    let bits = c.read_fixed32()?;
                    attr.floats.push(f32::from_bits(bits));
                }
            }
            _ => {
                c.skip_field(wire)?;
            }
        }
    }
    Ok(attr)
}

#[derive(Clone, Debug, Default)]
pub struct GraphNode {
    pub op_type: String,
    /// Operator domain ("" = default ONNX domain).
    pub domain: String,
    pub name: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: HashMap<String, Attribute>,
}

impl GraphNode {
    /// Operator key used for dispatch/reporting: `op_type` for the default
    /// domain, `domain::op_type` otherwise (so custom-domain ops never alias
    /// a default-domain kernel).
    pub fn op_key(&self) -> String {
        if self.domain.is_empty() || self.domain == "ai.onnx" {
            self.op_type.clone()
        } else {
            format!("{}::{}", self.domain, self.op_type)
        }
    }
}

fn parse_node(c: &mut Cursor) -> Result<GraphNode, String> {
    let mut node = GraphNode::default();
    while !c.eof() {
        let (field, wire) = c.read_tag()?;
        match field {
            1 => {
                node.inputs.push(String::from_utf8(c.read_length_delimited()?.to_vec())
                    .map_err(|_| "ONNX: invalid UTF-8 in node input name")?);
            }
            2 => {
                node.outputs.push(String::from_utf8(c.read_length_delimited()?.to_vec())
                    .map_err(|_| "ONNX: invalid UTF-8 in node output name")?);
            }
            3 => {
                node.name = String::from_utf8(c.read_length_delimited()?.to_vec())
                    .map_err(|_| "ONNX: invalid UTF-8 in node name")?;
            }
            4 => {
                node.op_type = String::from_utf8(c.read_length_delimited()?.to_vec())
                    .map_err(|_| "ONNX: invalid UTF-8 in op_type")?;
            }
            5 => {
                let bytes = c.read_length_delimited()?;
                let mut inner = Cursor::new(bytes);
                let attr = parse_attribute(&mut inner)?;
                node.attributes.insert(attr.name.clone(), attr);
            }
            7 => {
                node.domain = String::from_utf8(c.read_length_delimited()?.to_vec())
                    .map_err(|_| "ONNX: invalid UTF-8 in node domain")?;
            }
            _ => {
                c.skip_field(wire)?;
            }
        }
    }
    Ok(node)
}

struct TensorParseResult {
    name: String,
    data_type: i32,
    dims: Vec<i64>,
    float_data: Vec<f32>,
    int64_data: Vec<i64>,
}

// TensorProto field numbers: dims=1, data_type=2, float_data=4, int64_data=7,
// name=8, raw_data=9, external_data=13, data_location=14.
fn parse_tensor(c: &mut Cursor) -> Result<TensorParseResult, String> {
    let mut res = TensorParseResult {
        name: String::new(),
        data_type: 0,
        dims: Vec::new(),
        float_data: Vec::new(),
        int64_data: Vec::new(),
    };
    let mut raw: Option<&[u8]> = None;
    let mut external = false;
    while !c.eof() {
        let (field, wire) = c.read_tag()?;
        match field {
            1 => {
                if wire == 2 {
                    let bytes = c.read_length_delimited()?;
                    let mut inner = Cursor::new(bytes);
                    while !inner.eof() {
                        res.dims.push(inner.read_varint()? as i64);
                    }
                } else {
                    res.dims.push(c.read_varint()? as i64);
                }
            }
            2 => {
                res.data_type = c.read_varint()? as i32;
            }
            4 => {
                if wire == 2 {
                    let bytes = c.read_length_delimited()?;
                    let mut inner = Cursor::new(bytes);
                    while !inner.eof() {
                        res.float_data.push(f32::from_bits(inner.read_fixed32()?));
                    }
                } else {
                    res.float_data.push(f32::from_bits(c.read_fixed32()?));
                }
            }
            7 => {
                if wire == 2 {
                    let bytes = c.read_length_delimited()?;
                    let mut inner = Cursor::new(bytes);
                    while !inner.eof() {
                        res.int64_data.push(inner.read_varint()? as i64);
                    }
                } else {
                    res.int64_data.push(c.read_varint()? as i64);
                }
            }
            8 => {
                res.name = String::from_utf8(c.read_length_delimited()?.to_vec())
                    .map_err(|_| "ONNX: invalid UTF-8 in tensor name")?;
            }
            9 => {
                raw = Some(c.read_length_delimited()?);
            }
            13 => {
                external = true;
                c.skip_field(wire)?;
            }
            14 => {
                if c.read_varint()? == 1 {
                    external = true;
                }
            }
            _ => {
                c.skip_field(wire)?;
            }
        }
    }
    if external {
        return Err(format!(
            "ONNX: initializer '{}' uses external data, which the WASM runtime cannot load",
            res.name
        ));
    }
    // raw_data is decoded after the loop so field order does not matter.
    if let Some(bytes) = raw {
        match res.data_type {
            1 => {
                let (chunks, rest) = bytes.as_chunks::<4>();
                if !rest.is_empty() {
                    return Err("ONNX: float raw_data size not multiple of 4".into());
                }
                res.float_data = chunks.iter().map(|b| f32::from_le_bytes(*b)).collect();
            }
            7 => {
                let (chunks, rest) = bytes.as_chunks::<8>();
                if !rest.is_empty() {
                    return Err("ONNX: int64 raw_data size not multiple of 8".into());
                }
                res.int64_data = chunks.iter().map(|b| i64::from_le_bytes(*b)).collect();
            }
            _ => {}
        }
    }
    Ok(res)
}

/// Declared graph input/output: name, ONNX elem_type code, and shape
/// (`None` when no shape is declared; symbolic dims are reported as -1).
#[derive(Clone, Debug, Default)]
pub struct ValueInfo {
    pub name: String,
    pub elem_type: i32,
    pub shape: Option<Vec<i64>>,
}

fn parse_dimension(c: &mut Cursor) -> Result<i64, String> {
    let mut dim = -1i64;
    while !c.eof() {
        let (field, wire) = c.read_tag()?;
        if field == 1 {
            dim = c.read_varint()? as i64;
        } else {
            c.skip_field(wire)?;
        }
    }
    Ok(dim)
}

fn parse_tensor_type(c: &mut Cursor, vi: &mut ValueInfo) -> Result<(), String> {
    while !c.eof() {
        let (field, wire) = c.read_tag()?;
        match field {
            1 => vi.elem_type = c.read_varint()? as i32,
            2 => {
                let bytes = c.read_length_delimited()?;
                let mut shape_c = Cursor::new(bytes);
                let mut dims = Vec::new();
                while !shape_c.eof() {
                    let (f, w) = shape_c.read_tag()?;
                    if f == 1 {
                        let db = shape_c.read_length_delimited()?;
                        dims.push(parse_dimension(&mut Cursor::new(db))?);
                    } else {
                        shape_c.skip_field(w)?;
                    }
                }
                vi.shape = Some(dims);
            }
            _ => c.skip_field(wire)?,
        }
    }
    Ok(())
}

fn parse_value_info(c: &mut Cursor) -> Result<ValueInfo, String> {
    let mut vi = ValueInfo::default();
    while !c.eof() {
        let (field, wire) = c.read_tag()?;
        match field {
            1 => {
                vi.name = String::from_utf8(c.read_length_delimited()?.to_vec())
                    .map_err(|_| "ONNX: invalid UTF-8 in value info name")?;
            }
            2 => {
                // TypeProto: field 1 = tensor_type
                let bytes = c.read_length_delimited()?;
                let mut tc = Cursor::new(bytes);
                while !tc.eof() {
                    let (f, w) = tc.read_tag()?;
                    if f == 1 {
                        let tb = tc.read_length_delimited()?;
                        parse_tensor_type(&mut Cursor::new(tb), &mut vi)?;
                    } else {
                        tc.skip_field(w)?;
                    }
                }
            }
            _ => c.skip_field(wire)?,
        }
    }
    Ok(vi)
}

#[derive(Clone, Debug, Default)]
pub struct Graph {
    pub name: String,
    pub nodes: Vec<GraphNode>,
    pub weights: HashMap<String, Tensor>,
    pub int_initializers: HashMap<String, Vec<i64>>,
    /// Initializers of dtypes this runtime cannot compute with (name -> ONNX data_type).
    pub other_initializers: HashMap<String, i32>,
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
    pub inputs: Vec<ValueInfo>,
    pub outputs: Vec<ValueInfo>,
}

fn parse_graph(c: &mut Cursor) -> Result<Graph, String> {
    let mut g = Graph::default();
    while !c.eof() {
        let (field, wire) = c.read_tag()?;
        match field {
            1 => {
                let bytes = c.read_length_delimited()?;
                let mut inner = Cursor::new(bytes);
                g.nodes.push(parse_node(&mut inner)?);
            }
            2 => {
                g.name = String::from_utf8(c.read_length_delimited()?.to_vec())
                    .map_err(|_| "ONNX: invalid UTF-8 in graph name")?;
            }
            5 => {
                let bytes = c.read_length_delimited()?;
                let mut inner = Cursor::new(bytes);
                let tp = parse_tensor(&mut inner)?;
                if tp.name.is_empty() {
                    return Err("ONNX: initializer with empty name".into());
                }
                let expected_len = checked_numel(tp.dims.iter().copied(), "ONNX initializer")?;
                if tp.data_type == 1 {
                    if tp.float_data.len() != expected_len {
                        return Err(format!(
                            "ONNX: float initializer size mismatch (dims={:?}, data={})",
                            tp.dims,
                            tp.float_data.len()
                        ));
                    }
                    g.weights.insert(tp.name, Tensor::new(tp.dims, tp.float_data));
                } else if tp.data_type == 7 {
                    if tp.int64_data.len() != expected_len {
                        return Err(format!(
                            "ONNX: int64 initializer size mismatch (dims={:?}, expected={}, data={})",
                            tp.dims, expected_len, tp.int64_data.len()
                        ));
                    }
                    g.int_initializers.insert(tp.name, tp.int64_data);
                } else {
                    g.other_initializers.insert(tp.name, tp.data_type);
                }
            }
            11 => {
                let bytes = c.read_length_delimited()?;
                let mut inner = Cursor::new(bytes);
                let vi = parse_value_info(&mut inner)?;
                g.input_names.push(vi.name.clone());
                g.inputs.push(vi);
            }
            12 => {
                let bytes = c.read_length_delimited()?;
                let mut inner = Cursor::new(bytes);
                let vi = parse_value_info(&mut inner)?;
                g.output_names.push(vi.name.clone());
                g.outputs.push(vi);
            }
            _ => {
                c.skip_field(wire)?;
            }
        }
    }
    Ok(g)
}

#[derive(Clone, Debug, Default)]
pub struct Model {
    pub graph: Graph,
    /// Default-domain ("" or "ai.onnx") opset version; `None` if absent.
    pub opset: Option<i64>,
}

fn parse_opset_import(c: &mut Cursor) -> Result<(String, i64), String> {
    let mut domain = String::new();
    let mut version = 0i64;
    while !c.eof() {
        let (field, wire) = c.read_tag()?;
        match field {
            1 => {
                domain = String::from_utf8(c.read_length_delimited()?.to_vec())
                    .map_err(|_| "ONNX: invalid UTF-8 in opset domain")?;
            }
            2 => version = c.read_varint()? as i64,
            _ => c.skip_field(wire)?,
        }
    }
    Ok((domain, version))
}

fn parse_model(c: &mut Cursor) -> Result<Model, String> {
    let mut m = Model::default();
    while !c.eof() {
        let (field, wire) = c.read_tag()?;
        if field == 7 {
            let bytes = c.read_length_delimited()?;
            let mut inner = Cursor::new(bytes);
            m.graph = parse_graph(&mut inner)?;
        } else if field == 8 {
            let bytes = c.read_length_delimited()?;
            let (domain, version) = parse_opset_import(&mut Cursor::new(bytes))?;
            if domain.is_empty() || domain == "ai.onnx" {
                m.opset = Some(version);
            }
        } else {
            c.skip_field(wire)?;
        }
    }
    Ok(m)
}

/// Parse raw ONNX bytes into a [`Model`].
pub fn load_model(bytes: &[u8]) -> Result<Model, String> {
    parse_model(&mut Cursor::new(bytes))
}

// =================================────────────────===========================
// Topological Sort (Kahn's algorithm)
// =================================────────────────===========================

pub fn topological_sort(nodes: &[GraphNode]) -> Result<Vec<GraphNode>, String> {
    let mut producer = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        for out_name in &node.outputs {
            producer.entry(out_name.clone()).or_insert(i);
        }
    }

    let mut in_degree = vec![0; nodes.len()];
    let mut consumers = vec![vec![]; nodes.len()];

    for (i, node) in nodes.iter().enumerate() {
        for in_name in &node.inputs {
            if let Some(&p) = producer.get(in_name) {
                if p == i {
                    return Err(format!("topological_sort: self-loop on node '{}'", node.name));
                }
                in_degree[i] += 1;
                consumers[p].push(i);
            }
        }
    }

    let mut ready = std::collections::VecDeque::new();
    for (i, &deg) in in_degree.iter().enumerate() {
        if deg == 0 {
            ready.push_back(i);
        }
    }

    let mut sorted = Vec::with_capacity(nodes.len());
    while let Some(i) = ready.pop_front() {
        sorted.push(nodes[i].clone());
        for &c in &consumers[i] {
            in_degree[c] -= 1;
            if in_degree[c] == 0 {
                ready.push_back(c);
            }
        }
    }

    if sorted.len() != nodes.len() {
        return Err(format!(
            "topological_sort: cycle detected (sorted {} of {} nodes)",
            sorted.len(),
            nodes.len()
        ));
    }

    Ok(sorted)
}


// =================================────────────────===========================
// Pure-Rust Operator Subset Reimplementation
// ============================================================================

/// MatMul for rank-2 operands only (`[m,k] x [k,n] -> [m,n]`). Other ranks
/// are rejected with an error instead of being guessed at.
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(format!(
            "MatMul: WASM runtime only supports rank-2 operands, got A {:?} and B {:?}",
            a.shape, b.shape
        ));
    }
    let (m, k, k2, n) = (a.shape[0], a.shape[1], b.shape[0], b.shape[1]);
    if [m, k, n].iter().any(|&d| d < 0) {
        return Err(format!("MatMul: negative dimension in A {:?} / B {:?}", a.shape, b.shape));
    }
    if k != k2 {
        return Err(format!(
            "MatMul: inner dimensions differ (A {:?} has {} columns, B {:?} has {} rows)",
            a.shape, k, b.shape, k2
        ));
    }
    let (m, k, n) = (m as usize, k as usize, n as usize);
    let a_len = m.checked_mul(k).ok_or("MatMul: A size overflows usize")?;
    let b_len = k.checked_mul(n).ok_or("MatMul: B size overflows usize")?;
    if a.data.len() != a_len || b.data.len() != b_len {
        return Err(format!(
            "MatMul: shape/data mismatch (A {:?} len {}, B {:?} len {})",
            a.shape,
            a.data.len(),
            b.shape,
            b.data.len()
        ));
    }
    let total = checked_numel([m as i64, n as i64], "MatMul output")?;
    let mut data = vec![0.0f32; total];
    for r in 0..m {
        let a_row = &a.data[r * k..(r + 1) * k];
        let out_row = &mut data[r * n..(r + 1) * n];
        for (i, &av) in a_row.iter().enumerate() {
            let b_row = &b.data[i * n..(i + 1) * n];
            for (o, &bv) in out_row.iter_mut().zip(b_row) {
                *o += av * bv;
            }
        }
    }
    Ok(Tensor::new(vec![m as i64, n as i64], data))
}

pub fn relu(a: &Tensor) -> Tensor {
    let data = a.data.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect();
    Tensor::new(a.shape.clone(), data)
}

pub fn sigmoid(a: &Tensor) -> Tensor {
    let data = a.data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
    Tensor::new(a.shape.clone(), data)
}

/// Softmax following ONNX semantics for the given opset.
///
/// * opset >= 13 (or unknown): default axis = -1; normalises along `axis` only.
/// * opset < 13: default axis = 1; input is coerced to 2-D
///   `[prod(shape[..axis]), prod(shape[axis..])]` and each row is normalised.
pub fn softmax(input: &Tensor, axis_attr: Option<i64>, opset: Option<i64>) -> Result<Tensor, String> {
    let rank = input.shape.len();
    if rank == 0 {
        return Err("softmax: input must have rank >= 1".into());
    }
    let legacy = matches!(opset, Some(v) if v < 13);
    let axis_raw = axis_attr.unwrap_or(if legacy { 1 } else { -1 });
    let axis = if axis_raw < 0 { axis_raw + rank as i64 } else { axis_raw };
    // Legacy semantics allow axis == rank (inner block of size 1).
    let max_axis = if legacy { rank as i64 } else { rank as i64 - 1 };
    if axis < 0 || axis > max_axis {
        return Err(format!("softmax: axis {} out of range for rank {}", axis_raw, rank));
    }
    let axis = axis as usize;
    let dims: Vec<usize> = input.shape.iter().map(|&d| d.max(0) as usize).collect();
    let numel = |s: &[usize]| checked_numel(s.iter().map(|&d| d as i64), "softmax");

    // Normalise over `n` elements spaced by `stride`, in `outer * stride` groups.
    let (outer, n, stride) = if legacy {
        (numel(&dims[..axis])?, numel(&dims[axis..])?, 1usize)
    } else {
        (numel(&dims[..axis])?, dims[axis], numel(&dims[axis + 1..])?)
    };
    if checked_numel(input.shape.iter().copied(), "softmax")? != input.data.len()
        || outer.checked_mul(n).and_then(|v| v.checked_mul(stride)) != Some(input.data.len())
    {
        return Err(format!(
            "softmax: shape {:?} does not match data length {}",
            input.shape,
            input.data.len()
        ));
    }
    let mut data = vec![0.0f32; input.data.len()];
    if n == 0 {
        return Ok(Tensor::new(input.shape.clone(), data));
    }
    for o in 0..outer {
        for s in 0..stride {
            let base = o * n * stride + s;
            let idx = |j: usize| base + j * stride;
            let mut max_val = f32::NEG_INFINITY;
            for j in 0..n {
                max_val = max_val.max(input.data[idx(j)]);
            }
            let mut sum = 0.0f32;
            for j in 0..n {
                let e = (input.data[idx(j)] - max_val).exp();
                data[idx(j)] = e;
                sum += e;
            }
            for j in 0..n {
                data[idx(j)] /= sum;
            }
        }
    }
    Ok(Tensor::new(input.shape.clone(), data))
}

/// Elementwise Add with numpy-style multidirectional broadcasting.
pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    let rank = a.shape.len().max(b.shape.len());
    let pad = |s: &[i64]| -> Result<Vec<usize>, String> {
        let mut v = vec![1usize; rank - s.len()];
        for &d in s {
            v.push(usize::try_from(d).map_err(|_| format!("add: negative dimension in {:?}", s))?);
        }
        Ok(v)
    };
    let (sa, sb) = (pad(&a.shape)?, pad(&b.shape)?);
    let numel = |s: &[usize]| checked_numel(s.iter().map(|&d| d as i64), "add");
    if numel(&sa)? != a.data.len() || numel(&sb)? != b.data.len() {
        return Err(format!(
            "add: tensor shape/data mismatch (A {:?} len {}, B {:?} len {})",
            a.shape,
            a.data.len(),
            b.shape,
            b.data.len()
        ));
    }
    let mut out = vec![0usize; rank];
    for i in 0..rank {
        out[i] = if sa[i] == sb[i] || sb[i] == 1 {
            sa[i]
        } else if sa[i] == 1 {
            sb[i]
        } else {
            return Err(format!(
                "add: shapes {:?} and {:?} are not broadcast-compatible",
                a.shape, b.shape
            ));
        };
    }
    // Row-major strides; broadcast dims get stride 0.
    let strides = |s: &[usize]| -> Vec<usize> {
        let mut st = vec![0usize; rank];
        let mut acc = 1usize;
        for i in (0..rank).rev() {
            st[i] = if s[i] == 1 { 0 } else { acc };
            acc *= s[i];
        }
        st
    };
    let (sta, stb) = (strides(&sa), strides(&sb));
    let total = numel(&out)?;
    let mut data = Vec::with_capacity(total);
    let mut idx = vec![0usize; rank];
    let (mut ia, mut ib) = (0usize, 0usize);
    for _ in 0..total {
        data.push(a.data[ia] + b.data[ib]);
        for d in (0..rank).rev() {
            idx[d] += 1;
            ia += sta[d];
            ib += stb[d];
            if idx[d] < out[d] {
                break;
            }
            ia -= sta[d] * out[d];
            ib -= stb[d] * out[d];
            idx[d] = 0;
        }
    }
    Ok(Tensor::new(out.iter().map(|&d| d as i64).collect(), data))
}

// ============================================================================
// Pure-Rust Executor
// ============================================================================

/// Look up a node input: computed values first, then float initializers.
fn lookup<'a>(
    model: &'a Model,
    computed: &'a HashMap<String, Tensor>,
    name: &str,
) -> Result<&'a Tensor, String> {
    if let Some(t) = computed.get(name).or_else(|| model.graph.weights.get(name)) {
        return Ok(t);
    }
    if model.graph.int_initializers.contains_key(name) {
        return Err(format!("tensor '{}' is an int64 initializer; WASM runtime ops take float32 only", name));
    }
    if let Some(dt) = model.graph.other_initializers.get(name) {
        return Err(format!("tensor '{}' has ONNX data_type {}; WASM runtime ops take float32 only", name, dt));
    }
    Err(format!("tensor '{}' is neither an initializer nor a prior node's output", name))
}

/// Operators implemented by this WASM runtime.
pub const SUPPORTED_OPS: [&str; 5] = ["Add", "MatMul", "Relu", "Sigmoid", "Softmax"];

/// Graph inputs that are real runtime inputs (not initializers).
fn runtime_inputs(model: &Model) -> Vec<&ValueInfo> {
    let g = &model.graph;
    g.inputs
        .iter()
        .filter(|vi| {
            !g.weights.contains_key(&vi.name)
                && !g.int_initializers.contains_key(&vi.name)
                && !g.other_initializers.contains_key(&vi.name)
        })
        .collect()
}

/// One reason per distinct unsupported operator (empty = all supported).
fn unsupported_op_reasons(model: &Model) -> Vec<String> {
    let mut seen = std::collections::BTreeSet::new();
    model
        .graph
        .nodes
        .iter()
        .map(GraphNode::op_key)
        .filter(|k| !SUPPORTED_OPS.contains(&k.as_str()))
        .filter(|k| seen.insert(k.clone()))
        .map(|k| format!("Unsupported operator '{}' (WASM runtime supports: {})", k, SUPPORTED_OPS.join(", ")))
        .collect()
}

/// Static reasons the model cannot run here (empty = supported).
pub fn check_model(model: &Model) -> Vec<String> {
    let mut reasons = unsupported_op_reasons(model);
    let ins = runtime_inputs(model);
    if ins.len() != 1 {
        reasons.push(format!("WASM runtime requires exactly 1 graph input, model has {}", ins.len()));
    }
    for vi in ins {
        if vi.elem_type != 0 && vi.elem_type != 1 {
            reasons.push(format!("Input '{}' has elem_type {} (only float32 supported)", vi.name, vi.elem_type));
        }
    }
    if model.graph.outputs.is_empty() {
        reasons.push("Model has no outputs declared".into());
    }
    reasons
}

pub fn run_inference_internal(
    model: &Model,
    input_data: &[f32],
    input_shape: &[i64],
) -> Result<Tensor, String> {
    if let Some(first) = unsupported_op_reasons(model).into_iter().next() {
        return Err(first);
    }
    let ins = runtime_inputs(model);
    let input_vi = match ins.as_slice() {
        [one] => *one,
        _ => return Err(format!("WASM runtime requires exactly 1 graph input, model has {}", ins.len())),
    };
    if input_vi.elem_type != 0 && input_vi.elem_type != 1 {
        return Err(format!(
            "input '{}' has elem_type {} (only float32 supported)",
            input_vi.name, input_vi.elem_type
        ));
    }
    let expected = checked_numel(input_shape.iter().copied(), "input");
    if expected.as_ref().ok() != Some(&input_data.len()) {
        return Err(format!(
            "input data has {} elements but shape {:?} is invalid or needs a different count",
            input_data.len(),
            input_shape
        ));
    }
    if let Some(decl) = &input_vi.shape {
        let ok = decl.len() == input_shape.len()
            && decl.iter().zip(input_shape).all(|(&d, &s)| d <= 0 || d == s);
        if !ok {
            return Err(format!(
                "input '{}' shape {:?} does not match declared shape {:?} (-1 = any)",
                input_vi.name,
                input_shape,
                decl.iter().map(|&d| if d <= 0 { -1 } else { d }).collect::<Vec<_>>()
            ));
        }
    }

    // Computed values only; initializers are read in place via `lookup`.
    let mut computed: HashMap<String, Tensor> = HashMap::new();
    computed.insert(input_vi.name.clone(), Tensor::new(input_shape.to_vec(), input_data.to_vec()));

    let sorted_nodes = topological_sort(&model.graph.nodes)?;

    for node in &sorted_nodes {
        let ins = &node.inputs;
        let arity = if matches!(node.op_type.as_str(), "MatMul" | "Add") { 2 } else { 1 };
        if ins.len() != arity || node.outputs.len() != 1 {
            return Err(format!(
                "{} '{}': expected {} input(s) and 1 output, got {} and {}",
                node.op_type,
                node.name,
                arity,
                ins.len(),
                node.outputs.len()
            ));
        }
        let x = lookup(model, &computed, &ins[0])?;
        let y = match node.op_type.as_str() {
            "MatMul" => matmul(x, lookup(model, &computed, &ins[1])?)?,
            "Add" => add(x, lookup(model, &computed, &ins[1])?)?,
            "Relu" => relu(x),
            "Sigmoid" => sigmoid(x),
            "Softmax" => softmax(x, node.attributes.get("axis").map(|a| a.i), model.opset)?,
            other => {
                return Err(format!(
                    "Unsupported operator '{}' (WASM runtime supports: {})",
                    other,
                    SUPPORTED_OPS.join(", ")
                ))
            }
        };
        computed.insert(node.outputs[0].clone(), y);
    }

    let output_name = model
        .graph
        .output_names
        .first()
        .ok_or_else(|| "Model has no outputs declared".to_string())?;
    match computed.remove(output_name) {
        Some(t) => Ok(t),
        None => lookup(model, &computed, output_name).cloned(),
    }
}

fn json_str(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for ch in s.chars() {
        match ch {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn json_value_infos(vis: &[&ValueInfo]) -> String {
    let items: Vec<String> = vis
        .iter()
        .map(|vi| {
            let shape = match &vi.shape {
                Some(s) => format!(
                    "[{}]",
                    s.iter().map(|&d| if d <= 0 { -1 } else { d }.to_string()).collect::<Vec<_>>().join(",")
                ),
                None => "null".into(),
            };
            format!("{{\"name\":{},\"elem_type\":{},\"shape\":{}}}", json_str(&vi.name), vi.elem_type, shape)
        })
        .collect();
    format!("[{}]", items.join(","))
}

/// JSON metadata for a parsed model (see `modelInfo` export).
pub fn model_info_json(model: &Model) -> String {
    let mut ops: Vec<String> = model.graph.nodes.iter().map(GraphNode::op_key).collect();
    ops.sort_unstable();
    ops.dedup();
    let reasons = check_model(model);
    let outputs: Vec<&ValueInfo> = model.graph.outputs.iter().collect();
    format!(
        "{{\"opset\":{},\"num_nodes\":{},\"inputs\":{},\"outputs\":{},\"ops_used\":[{}],\"supported\":{},\"unsupported_reasons\":[{}]}}",
        model.opset.map_or("null".to_string(), |v| v.to_string()),
        model.graph.nodes.len(),
        json_value_infos(&runtime_inputs(model)),
        json_value_infos(&outputs),
        ops.iter().map(|s| json_str(s)).collect::<Vec<_>>().join(","),
        reasons.is_empty(),
        reasons.iter().map(|s| json_str(s)).collect::<Vec<_>>().join(",")
    )
}

// ============================================================================
// Public JS FFI interface
// ============================================================================

/// Runs the model on one float32 input and returns the first output's data
/// (row-major). Throws a string on unsupported ops or bad shapes.
#[wasm_bindgen(js_name = runInference)]
pub fn run_inference(
    model_bytes: &[u8],
    input_data: &[f32],
    input_shape: Vec<i32>,
) -> Result<Vec<f32>, JsValue> {
    let model = load_model(model_bytes).map_err(|e| JsValue::from_str(&e))?;
    let shape_i64: Vec<i64> = input_shape.iter().map(|&x| x as i64).collect();
    let output = run_inference_internal(&model, input_data, &shape_i64)
        .map_err(|e| JsValue::from_str(&e))?;
    Ok(output.data)
}

/// Runs the model like `runInference` and returns only the output shape
/// (row-major dims of the first graph output).
#[wasm_bindgen(js_name = outputShape)]
pub fn output_shape(
    model_bytes: &[u8],
    input_data: &[f32],
    input_shape: Vec<i32>,
) -> Result<Vec<i32>, JsValue> {
    let model = load_model(model_bytes).map_err(|e| JsValue::from_str(&e))?;
    let shape_i64: Vec<i64> = input_shape.iter().map(|&x| x as i64).collect();
    let output = run_inference_internal(&model, input_data, &shape_i64)
        .map_err(|e| JsValue::from_str(&e))?;
    Ok(output.shape.iter().map(|&d| d as i32).collect())
}

/// Operators this WASM runtime implements.
#[wasm_bindgen(js_name = supportedOps)]
pub fn supported_ops() -> Vec<String> {
    SUPPORTED_OPS.iter().map(|s| s.to_string()).collect()
}

/// Model metadata as a JSON string (parse with `JSON.parse`).
#[wasm_bindgen(js_name = modelInfo)]
pub fn model_info(model_bytes: &[u8]) -> Result<String, JsValue> {
    let model = load_model(model_bytes).map_err(|e| JsValue::from_str(&e))?;
    Ok(model_info_json(&model))
}

// ============================================================================
// Typed fraud-detection convenience wrapper
// ============================================================================
//
// Normalisation constants baked in from models/fraud/model_config.json.
// Input features (7): amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest,
//   newbalanceDest, type_CASH_OUT (0/1), type_TRANSFER (0/1)
// Output: fraud probability in [0, 1]

const FRAUD_MEAN: [f32; 7] = [
    11948.5234375,
    86466.6171875,
    74708.890625,
    59117.32421875,
    71066.4609375,
    0.34942499,
    0.15512501,
];

const FRAUD_STD: [f32; 7] = [
    68235.4140625,
    100729.7578125,
    80026.859375,
    59793.71875,
    86849.28125,
    0.47677723,
    0.36204198,
];

/// Fraud probability for one raw (un-normalised) 7-feature transaction.
pub fn fraud_score(model_bytes: &[u8], raw: [f32; 7]) -> Result<f32, String> {
    let normalised: Vec<f32> = raw
        .iter()
        .zip(FRAUD_MEAN.iter().zip(FRAUD_STD.iter()))
        .map(|(&v, (&mean, &std))| (v - mean) / std)
        .collect();
    let model = load_model(model_bytes)?;
    let output = run_inference_internal(&model, &normalised, &[1, 7])?;
    match output.data.as_slice() {
        [p] => Ok(*p),
        other => Err(format!("runFraudModel: expected 1 output value, got {}", other.len())),
    }
}

/// Fraud probability for one raw transaction: z-score normalises the seven
/// features with the training constants, then runs the model on `[1, 7]`.
#[wasm_bindgen(js_name = runFraudModel)]
pub fn run_fraud_model(
    model_bytes: &[u8],
    amount: f32,
    old_balance_orig: f32,
    new_balance_orig: f32,
    old_balance_dest: f32,
    new_balance_dest: f32,
    is_cash_out: f32,
    is_transfer: f32,
) -> Result<f32, JsValue> {
    fraud_score(
        model_bytes,
        [
            amount,
            old_balance_orig,
            new_balance_orig,
            old_balance_dest,
            new_balance_dest,
            is_cash_out,
            is_transfer,
        ],
    )
    .map_err(|e| JsValue::from_str(&e))
}

// ============================================================================
// Unit Tests (ORT parity tests live in tests/parity.rs)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn repo_file(rel: &str) -> Vec<u8> {
        let p = format!("{}/../{}", env!("CARGO_MANIFEST_DIR"), rel);
        std::fs::read(&p).unwrap_or_else(|e| panic!("read {p}: {e}"))
    }

    fn t(shape: &[i64], data: &[f32]) -> Tensor {
        Tensor::new(shape.to_vec(), data.to_vec())
    }

    #[test]
    fn test_empty_onnx_parse() {
        // empty.onnx is the smallest legal model containing no nodes.
        let bytes = repo_file("engine/tests/fixtures/empty.onnx");
        let model = load_model(&bytes).unwrap();
        assert_eq!(model.graph.name, "empty");
        assert_eq!(model.graph.nodes.len(), 0);
        assert_eq!(model.graph.input_names, vec!["X"]);
        assert_eq!(model.graph.output_names, vec!["Y"]);
    }

    #[test]
    fn test_matmul_add_onnx_parse() {
        let bytes = repo_file("engine/tests/fixtures/matmul_add.onnx");
        let model = load_model(&bytes).unwrap();
        assert!(model.opset.is_some());
        assert_eq!(model.graph.name, "matmul_add");
        assert_eq!(model.graph.nodes.len(), 2);
        assert_eq!(model.graph.nodes[0].op_type, "MatMul");
        assert_eq!(model.graph.nodes[1].op_type, "Add");
        assert_eq!(model.graph.input_names, vec!["X"]);
        assert_eq!(model.graph.output_names, vec!["Y"]);
        assert!(model.graph.weights.contains_key("W"));
        assert!(model.graph.weights.contains_key("b"));
    }

    #[test]
    fn test_relu() {
        let x = Tensor::new(vec![4], vec![-2.0, -1.0, 0.0, 3.0]);
        let y = relu(&x);
        assert_eq!(y.data, vec![0.0, 0.0, 0.0, 3.0]);
    }

    #[test]
    fn test_sigmoid() {
        let x = Tensor::new(vec![1], vec![0.0]);
        let y = sigmoid(&x);
        assert_eq!(y.data, vec![0.5]);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let x = t(&[3], &[1.0, 2.0, 3.0]);
        let y = softmax(&x, None, Some(13)).unwrap();
        let sum: f32 = y.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_axis_semantics() {
        // [1,2,2]: opset 13 default axis=-1 normalises pairs along the last dim.
        let x = t(&[1, 2, 2], &[0.0, 0.0, 1.0, 1.0]);
        let y13 = softmax(&x, None, Some(13)).unwrap();
        assert_eq!(y13.data, vec![0.5, 0.5, 0.5, 0.5]);
        // opset 11 default axis=1 coerces to [1,4]: all four share one denominator.
        let y11 = softmax(&x, None, Some(11)).unwrap();
        let e = 1f32.exp();
        let d = 2.0 + 2.0 * e;
        for (got, want) in y11.data.iter().zip([1.0 / d, 1.0 / d, e / d, e / d]) {
            assert!((got - want).abs() < 1e-6);
        }
        // opset 13 axis=1 is strided (normalise column-wise).
        let y = softmax(&x, Some(1), Some(13)).unwrap();
        assert!((y.data[0] + y.data[2] - 1.0).abs() < 1e-6);
        assert!(softmax(&x, Some(3), Some(13)).is_err());
    }

    #[test]
    fn test_add_broadcasting() {
        let a = t(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let y = add(&a, &t(&[2, 2], &[0.5, 1.5, 2.5, 3.5])).unwrap();
        assert_eq!(y.data, vec![1.5, 3.5, 5.5, 7.5]);

        // Trailing 1-D broadcast (previously produced zeros).
        let y = add(&a, &t(&[2], &[10.0, 20.0])).unwrap();
        assert_eq!((y.shape.clone(), y.data), (vec![2, 2], vec![11.0, 22.0, 13.0, 24.0]));

        // Column vector [2,1] broadcasts across columns.
        let y = add(&a, &t(&[2, 1], &[10.0, 20.0])).unwrap();
        assert_eq!(y.data, vec![11.0, 12.0, 23.0, 24.0]);

        // Both-direction [2,1] + [1,3] -> [2,3].
        let y = add(&t(&[2, 1], &[1.0, 2.0]), &t(&[1, 3], &[10.0, 20.0, 30.0])).unwrap();
        assert_eq!((y.shape.clone(), y.data), (vec![2, 3], vec![11.0, 21.0, 31.0, 12.0, 22.0, 32.0]));

        // Rank-0 scalar.
        let y = add(&a, &t(&[], &[1.0])).unwrap();
        assert_eq!(y.data, vec![2.0, 3.0, 4.0, 5.0]);

        // Incompatible shapes are an error, never zeros.
        let err = add(&a, &t(&[3], &[1.0, 2.0, 3.0])).unwrap_err();
        assert!(err.contains("not broadcast-compatible"), "{err}");
        // Shape/data mismatch is an error, never a panic.
        assert!(add(&a, &t(&[3], &[1.0])).is_err());
    }

    #[test]
    fn test_matmul_shape_checks() {
        let a = t(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = t(&[3, 1], &[1.0, 1.0, 1.0]);
        let y = matmul(&a, &b).unwrap();
        assert_eq!((y.shape, y.data), (vec![2, 1], vec![6.0, 15.0]));
        assert!(matmul(&a, &a).unwrap_err().contains("inner dimensions differ"));
        assert!(matmul(&t(&[3], &[1.0, 2.0, 3.0]), &b).unwrap_err().contains("rank-2"));
        assert!(matmul(&a, &t(&[3, 1], &[1.0])).unwrap_err().contains("shape/data mismatch"));
    }

    #[test]
    fn test_huge_shapes_rejected() {
        // [65536,1] + [1,65536] broadcasts to 4G elements: must Err, not OOM.
        let a = t(&[65536, 1], &vec![0.0; 65536]);
        let b = t(&[1, 65536], &vec![0.0; 65536]);
        assert!(add(&a, &b).unwrap_err().contains("element limit"));
        // Products that overflow usize.
        let big = t(&[i64::MAX, i64::MAX, 0], &[]);
        assert!(add(&big, &big).is_err());
        assert!(softmax(&big, None, Some(13)).is_err());
        assert!(add(&t(&[-1], &[]), &t(&[1], &[1.0])).is_err());
        // MatMul output over the cap with tiny (k = 0) operands.
        let a = t(&[1 << 20, 0], &[]);
        let b = t(&[0, 1 << 20], &[]);
        assert!(matmul(&a, &b).unwrap_err().contains("element limit"));
        // Runtime input shape over the cap.
        let model = load_model(&repo_file("models/fraud/fraud_detector.onnx")).unwrap();
        assert!(run_inference_internal(&model, &[], &[1 << 40, 1 << 40, 0]).is_err());
    }

    /// Minimal protobuf encoders for hand-built malformed models.
    fn varint(mut v: u64, out: &mut Vec<u8>) {
        while v >= 0x80 {
            out.push((v as u8) | 0x80);
            v >>= 7;
        }
        out.push(v as u8);
    }
    fn ld(field: u64, payload: &[u8]) -> Vec<u8> {
        let mut o = Vec::new();
        varint(field << 3 | 2, &mut o);
        varint(payload.len() as u64, &mut o);
        o.extend_from_slice(payload);
        o
    }

    #[test]
    fn test_malformed_protobuf_errors_not_panics() {
        // Initializer claiming dims [2^31, 2^31] with 1 float of data.
        let mut tensor = Vec::new();
        for d in [1u64 << 31, 1 << 31] {
            varint(1 << 3, &mut tensor);
            varint(d, &mut tensor);
        }
        varint(2 << 3, &mut tensor);
        varint(1, &mut tensor);
        tensor.extend(ld(8, b"W"));
        tensor.extend(ld(9, &[0, 0, 0, 0]));
        let huge_init = ld(7, &ld(5, &tensor));
        // Negative dim (varint of u64::MAX = -1).
        let mut neg = Vec::new();
        varint(1 << 3, &mut neg);
        varint(u64::MAX, &mut neg);
        neg.extend(ld(8, b"N"));
        let neg_init = ld(7, &ld(5, &neg));

        let mut huge_len = Vec::new();
        varint(7 << 3 | 2, &mut huge_len);
        varint(u64::MAX, &mut huge_len);
        let mut huge_skip = Vec::new();
        varint(99 << 3 | 2, &mut huge_skip);
        varint(1 << 40, &mut huge_skip);

        let cases: Vec<(&str, Vec<u8>)> = vec![
            ("truncated varint", vec![0x08, 0x80]),
            ("overlong varint", vec![0x08, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01]),
            ("field 0", vec![0x00]),
            ("bad wire type", vec![0x0f]),
            ("truncated fixed32", vec![0x0d, 0x01]),
            ("truncated fixed64", vec![0x09, 0x01, 0x02]),
            ("huge length", huge_len),
            ("huge skip", huge_skip),
            ("truncated graph", ld(7, &[0x2a, 0x05, 0x01])),
            ("huge initializer dims", huge_init),
            ("negative initializer dim", neg_init),
        ];
        for (name, bytes) in &cases {
            assert!(load_model(bytes).is_err(), "{name} parsed");
        }
        // Every truncation of a real model must Err or parse, never panic.
        let good = repo_file("models/fraud/fraud_detector.onnx");
        for n in 0..good.len() {
            let _ = load_model(&good[..n]);
        }
    }

    #[test]
    fn test_supported_ops_and_model_info() {
        assert_eq!(supported_ops(), vec!["Add", "MatMul", "Relu", "Sigmoid", "Softmax"]);
        let model = load_model(&repo_file("models/fraud/fraud_detector.onnx")).unwrap();
        let info = model_info_json(&model);
        assert!(info.starts_with("{\"opset\":13,\"num_nodes\":3,"), "{info}");
        assert!(info.contains("\"inputs\":[{\"name\":\"input\",\"elem_type\":1,\"shape\":[-1,7]}]"), "{info}");
        assert!(info.contains("\"ops_used\":[\"Add\",\"MatMul\",\"Sigmoid\"]"), "{info}");
        assert!(info.contains("\"supported\":true,\"unsupported_reasons\":[]"), "{info}");
    }

    #[test]
    fn test_extreme_fraud_fuzzing() {
        let bytes = repo_file("web/public/models/fraud_detector.onnx");

        let mut seed: u32 = 1337;
        let mut next_random = || {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            (seed % 1000) as f32 / 10.0
        };

        for i in 0..2000 {
            let amount = next_random() * 1000.0;
            let old_balance_orig = next_random() * 5000.0;
            let new_balance_orig = (old_balance_orig - amount).max(0.0);
            let old_balance_dest = next_random() * 5000.0;
            let new_balance_dest = old_balance_dest + amount;
            let is_cash_out = if i % 2 == 0 { 1.0 } else { 0.0 };
            let is_transfer = if i % 2 != 0 { 1.0 } else { 0.0 };

            let val = fraud_score(
                &bytes,
                [
                    amount,
                    old_balance_orig,
                    new_balance_orig,
                    old_balance_dest,
                    new_balance_dest,
                    is_cash_out,
                    is_transfer,
                ],
            )
            .unwrap_or_else(|e| panic!("Fuzz iteration {i} failed: {e}"));
            assert!((0.0..=1.0).contains(&val), "Fuzz iteration {i} returned invalid score: {val}");
        }
    }
}
