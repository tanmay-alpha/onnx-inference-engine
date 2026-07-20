#![allow(clippy::excessive_precision, clippy::too_many_arguments)]

use wasm_bindgen::prelude::*;
use std::collections::HashMap;

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
        let len = self.read_varint()? as usize;
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
                let len = self.read_varint()? as usize;
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
            7 => {
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
            8 => {
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
    pub name: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: HashMap<String, Attribute>,
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

fn parse_tensor(c: &mut Cursor) -> Result<TensorParseResult, String> {
    let mut res = TensorParseResult {
        name: String::new(),
        data_type: 0,
        dims: Vec::new(),
        float_data: Vec::new(),
        int64_data: Vec::new(),
    };
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
            8 => {
                res.name = String::from_utf8(c.read_length_delimited()?.to_vec())
                    .map_err(|_| "ONNX: invalid UTF-8 in tensor name")?;
            }
            9 => {
                let bytes = c.read_length_delimited()?;
                if res.data_type == 1 {
                    if bytes.len() % 4 != 0 {
                        return Err("ONNX: float raw_data size not multiple of 4".into());
                    }
                    let count = bytes.len() / 4;
                    let mut data = vec![0.0f32; count];
                    for i in 0..count {
                        let mut b = [0u8; 4];
                        b.copy_from_slice(&bytes[i * 4..i * 4 + 4]);
                        data[i] = f32::from_le_bytes(b);
                    }
                    res.float_data = data;
                } else if res.data_type == 7 {
                    if bytes.len() % 8 != 0 {
                        return Err("ONNX: int64 raw_data size not multiple of 8".into());
                    }
                    let count = bytes.len() / 8;
                    let mut data = vec![0i64; count];
                    for i in 0..count {
                        let mut b = [0u8; 8];
                        b.copy_from_slice(&bytes[i * 8..i * 8 + 8]);
                        data[i] = i64::from_le_bytes(b);
                    }
                    res.int64_data = data;
                }
            }
            13 => {
                if wire == 2 {
                    let bytes = c.read_length_delimited()?;
                    let mut inner = Cursor::new(bytes);
                    while !inner.eof() {
                        let bits = inner.read_fixed32()?;
                        res.float_data.push(f32::from_bits(bits));
                    }
                } else {
                    let bits = c.read_fixed32()?;
                    res.float_data.push(f32::from_bits(bits));
                }
            }
            _ => {
                c.skip_field(wire)?;
            }
        }
    }
    Ok(res)
}

fn parse_value_info(c: &mut Cursor) -> Result<String, String> {
    let mut name = String::new();
    while !c.eof() {
        let (field, wire) = c.read_tag()?;
        if field == 1 {
            name = String::from_utf8(c.read_length_delimited()?.to_vec())
                .map_err(|_| "ONNX: invalid UTF-8 in value info name")?;
        } else {
            c.skip_field(wire)?;
        }
    }
    Ok(name)
}

#[derive(Clone, Debug, Default)]
pub struct Graph {
    pub name: String,
    pub nodes: Vec<GraphNode>,
    pub weights: HashMap<String, Tensor>,
    pub int_initializers: HashMap<String, Vec<i64>>,
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
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
                if tp.data_type == 1 {
                    let expected_len = if tp.dims.is_empty() {
                        1
                    } else {
                        tp.dims.iter().try_fold(1usize, |acc, &d| {
                            if d <= 0 {
                                return Err("ONNX: non-positive dimension".to_string());
                            }
                            let d_us = d as usize;
                            acc.checked_mul(d_us)
                                .ok_or_else(|| "ONNX: dim product overflowed usize".to_string())
                        })?
                    };
                    if tp.float_data.len() != expected_len {
                        return Err(format!(
                            "ONNX: float initializer size mismatch (dims={:?}, data={})",
                            tp.dims,
                            tp.float_data.len()
                        ));
                    }
                    g.weights.insert(tp.name, Tensor::new(tp.dims, tp.float_data));
                } else if tp.data_type == 7 {
                    g.int_initializers.insert(tp.name, tp.int64_data);
                }
            }
            11 => {
                let bytes = c.read_length_delimited()?;
                let mut inner = Cursor::new(bytes);
                g.input_names.push(parse_value_info(&mut inner)?);
            }
            12 => {
                let bytes = c.read_length_delimited()?;
                let mut inner = Cursor::new(bytes);
                g.output_names.push(parse_value_info(&mut inner)?);
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
}

fn parse_model(c: &mut Cursor) -> Result<Model, String> {
    let mut m = Model::default();
    while !c.eof() {
        let (field, wire) = c.read_tag()?;
        if field == 7 {
            let bytes = c.read_length_delimited()?;
            let mut inner = Cursor::new(bytes);
            m.graph = parse_graph(&mut inner)?;
        } else {
            c.skip_field(wire)?;
        }
    }
    Ok(m)
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

    let mut in_degree = vec![0; nodes.size()];
    let mut consumers = vec![vec![]; nodes.size()];

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

// Helper trait to query size/len of slices
trait SizeExt {
    fn size(&self) -> usize;
}
impl<T> SizeExt for [T] {
    fn size(&self) -> usize {
        self.len()
    }
}

// =================================────────────────===========================
// Pure-Rust Operator Subset Reimplementation
// ============================================================================

pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    if a.shape.len() != 2 {
        return Err(format!("matmul A: expected 2-D tensor, got rank {}", a.shape.len()));
    }
    if b.shape.len() != 2 {
        return Err(format!("matmul B: expected 2-D tensor, got rank {}", b.shape.len()));
    }
    let m = a.shape[0];
    let k1 = a.shape[1];
    let k2 = b.shape[0];
    let n = b.shape[1];
    if k1 != k2 {
        return Err(format!("matmul: inner dimension mismatch (A.cols={}, B.rows={})", k1, k2));
    }

    let m_us = m as usize;
    let n_us = n as usize;
    let k1_us = k1 as usize;
    let total = m_us.checked_mul(n_us).ok_or_else(|| {
        format!("matmul: output shape {}x{} overflows usize", m, n)
    })?;
    let mut data = vec![0.0f32; total];
    for r in 0..m_us {
        for c in 0..n_us {
            let mut sum = 0.0f32;
            for i in 0..k1_us {
                sum += a.data[r * k1_us + i] * b.data[i * n_us + c];
            }
            data[r * n_us + c] = sum;
        }
    }
    Ok(Tensor::new(vec![m, n], data))
}

pub fn relu(a: &Tensor) -> Tensor {
    let data = a.data.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect();
    Tensor::new(a.shape.clone(), data)
}

pub fn sigmoid(a: &Tensor) -> Tensor {
    let data = a.data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
    Tensor::new(a.shape.clone(), data)
}

pub fn softmax(input: &Tensor, axis_attr: Option<i64>) -> Result<Tensor, String> {
    if input.shape.is_empty() {
        return Err("softmax: input must have rank >= 1".into());
    }
    let rank = input.shape.len();
    let axis_attr = axis_attr.unwrap_or(-1);
    let mut axis = axis_attr;
    if axis < 0 {
        axis += rank as i64;
    }
    if axis < 0 || axis >= rank as i64 {
        return Err(format!("softmax: axis {} out of range for rank {}", axis_attr, rank));
    }

    if axis != (rank - 1) as i64 {
        return Err(format!("softmax: only axis = last dim (axis=-1) is supported, got {}", axis));
    }

    let axis_dim = input.shape[axis as usize] as usize;
    let outer = input.data.len() / axis_dim;
    if outer == 0 {
        return Ok(Tensor::new(input.shape.clone(), vec![0.0; input.data.len()]));
    }

    let mut data = vec![0.0f32; input.data.len()];
    for i in 0..outer {
        let start = i * axis_dim;
        let end = start + axis_dim;
        let slice = &input.data[start..end];

        let mut max_val = slice[0];
        for &v in slice.iter().skip(1) {
            if v > max_val {
                max_val = v;
            }
        }

        let mut exps = vec![0.0f32; axis_dim];
        let mut sum = 0.0f32;
        for j in 0..axis_dim {
            let val = (slice[j] - max_val).exp();
            exps[j] = val;
            sum += val;
        }

            if sum == 0.0 {
                let uniform = 1.0 / axis_dim as f32;
                for j in 0..axis_dim {
                    data[start + j] = uniform;
                }
            } else {
                for j in 0..axis_dim {
                    data[start + j] = exps[j] / sum;
                }
            }
    }

    Ok(Tensor::new(input.shape.clone(), data))
}


pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    if a.shape == b.shape {
        let mut data = Vec::with_capacity(a.data.len());
        for i in 0..a.data.len() {
            data.push(a.data[i] + b.data[i]);
        }
        return Ok(Tensor::new(a.shape.clone(), data));
    }

    if b.data.len() == 1 {
        let val = b.data[0];
        let data = a.data.iter().map(|&x| x + val).collect();
        return Ok(Tensor::new(a.shape.clone(), data));
    }
    if a.data.len() == 1 {
        let val = a.data[0];
        let data = b.data.iter().map(|&x| x + val).collect();
        return Ok(Tensor::new(b.shape.clone(), data));
    }

    if b.shape.len() == 1 && !a.shape.is_empty() && b.shape[0] == *a.shape.last().unwrap() {
        let c = b.shape[0] as usize;
        let spatial = a.data.len() / c / (a.shape[0] as usize);
        let n_outer = a.shape[0] as usize;
        let mut data = vec![0.0f32; a.data.len()];
        for n in 0..n_outer {
            for j in 0..c {
                let val_b = b.data[j];
                let base = (n * c + j) * spatial;
                for k in 0..spatial {
                    data[base + k] = a.data[base + k] + val_b;
                }
            }
        }
        return Ok(Tensor::new(a.shape.clone(), data));
    }

    Err(format!(
        "add: unsupported broadcast (A shape {:?} B shape {:?})",
        a.shape, b.shape
    ))
}

// ============================================================================
// Pure-Rust Executor
// ============================================================================

fn require_tensor<'a>(map: &'a HashMap<String, Tensor>, name: &str) -> Result<&'a Tensor, String> {
    map.get(name).ok_or_else(|| {
        format!(
            "run_inference: tensor '{}' is neither an initializer nor a prior node's output",
            name
        )
    })
}

fn run_inference_internal(
    model: &Model,
    input_data: &[f32],
    input_shape: &[i64],
) -> Result<Tensor, String> {
    if model.graph.input_names.is_empty() {
        return Err("Model has no inputs declared".into());
    }
    let input_name = &model.graph.input_names[0];

    let mut tensor_map = HashMap::new();
    for (k, v) in &model.graph.weights {
        tensor_map.insert(k.clone(), v.clone());
    }

    let input_tensor = Tensor::new(input_shape.to_vec(), input_data.to_vec());
    tensor_map.insert(input_name.clone(), input_tensor);

    let sorted_nodes = topological_sort(&model.graph.nodes)?;

    for node in &sorted_nodes {
        let op = &node.op_type;
        let ins = &node.inputs;
        let outs = &node.outputs;

        if op == "MatMul" {
            if ins.len() < 2 {
                return Err(format!("MatMul '{}': need 2 inputs, got {}", node.name, ins.len()));
            }
            let a = require_tensor(&tensor_map, &ins[0])?;
            let b = require_tensor(&tensor_map, &ins[1])?;
            let y = matmul(a, b)?;
            for out in outs {
                tensor_map.insert(out.clone(), y.clone());
            }
        } else if op == "Relu" {
            if ins.is_empty() {
                return Err(format!("Relu '{}': need 1 input, got 0", node.name));
            }
            let x = require_tensor(&tensor_map, &ins[0])?;
            let y = relu(x);
            for out in outs {
                tensor_map.insert(out.clone(), y.clone());
            }
        } else if op == "Sigmoid" {
            if ins.is_empty() {
                return Err(format!("Sigmoid '{}': need 1 input, got 0", node.name));
            }
            let x = require_tensor(&tensor_map, &ins[0])?;
            let y = sigmoid(x);
            for out in outs {
                tensor_map.insert(out.clone(), y.clone());
            }
        } else if op == "Softmax" {
            if ins.is_empty() {
                return Err(format!("Softmax '{}': need 1 input, got 0", node.name));
            }
            let x = require_tensor(&tensor_map, &ins[0])?;
            let axis_attr = node.attributes.get("axis").map(|a| a.i);
            let y = softmax(x, axis_attr)?;
            for out in outs {
                tensor_map.insert(out.clone(), y.clone());
            }
        } else if op == "Add" {
            if ins.len() < 2 {
                return Err(format!("Add '{}': need 2 inputs, got {}", node.name, ins.len()));
            }
            let a = require_tensor(&tensor_map, &ins[0])?;
            let b = require_tensor(&tensor_map, &ins[1])?;
            let y = add(a, b)?;
            for out in outs {
                tensor_map.insert(out.clone(), y.clone());
            }
        } else {
            return Err(format!("Unsupported operator '{}' in WASM runtime", op));
        }
    }

    if model.graph.output_names.is_empty() {
        return Err("Model has no outputs declared".into());
    }
    let output_name = &model.graph.output_names[0];
    let output_tensor = require_tensor(&tensor_map, output_name)?;
    Ok(output_tensor.clone())
}

// ============================================================================
// Public JS FFI interface
// ============================================================================

#[wasm_bindgen(js_name = runInference)]
pub fn run_inference(
    model_bytes: &[u8],
    input_data: &[f32],
    input_shape: Vec<i32>,
) -> Result<Vec<f32>, JsValue> {
    let mut cursor = Cursor::new(model_bytes);
    let model = parse_model(&mut cursor).map_err(|e| JsValue::from_str(&e))?;

    let shape_i64: Vec<i64> = input_shape.iter().map(|&x| x as i64).collect();
    let output = run_inference_internal(&model, input_data, &shape_i64)
        .map_err(|e| JsValue::from_str(&e))?;

    Ok(output.data)
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
    let raw = [
        amount,
        old_balance_orig,
        new_balance_orig,
        old_balance_dest,
        new_balance_dest,
        is_cash_out,
        is_transfer,
    ];

    // Z-score normalise
    let normalised: Vec<f32> = raw
        .iter()
        .enumerate()
        .map(|(i, &v)| (v - FRAUD_MEAN[i]) / FRAUD_STD[i])
        .collect();

    let mut cursor = Cursor::new(model_bytes);
    let model = parse_model(&mut cursor).map_err(|e| JsValue::from_str(&e))?;

    let output = run_inference_internal(&model, &normalised, &[1, 7])
        .map_err(|e| JsValue::from_str(&e))?;

    // output is shape [1,1]; return the single probability value
    output
        .data
        .first()
        .copied()
        .ok_or_else(|| JsValue::from_str("run_fraud_model: empty output tensor"))
}


// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_onnx_parse() {
        // empty.onnx is the smallest legal model containing no nodes.
        let bytes = std::fs::read("../engine/tests/fixtures/empty.onnx").unwrap();
        let mut cursor = Cursor::new(&bytes);
        let model = parse_model(&mut cursor).unwrap();
        assert_eq!(model.graph.name, "empty");
        assert_eq!(model.graph.nodes.len(), 0);
        assert_eq!(model.graph.input_names, vec!["X"]);
        assert_eq!(model.graph.output_names, vec!["Y"]);
    }

    #[test]
    fn test_matmul_add_onnx_parse() {
        let bytes = std::fs::read("../engine/tests/fixtures/matmul_add.onnx").unwrap();
        let mut cursor = Cursor::new(&bytes);
        let model = parse_model(&mut cursor).unwrap();
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
    fn test_softmax() {
        let x = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]);
        let y = softmax(&x, None).unwrap();
        let sum: f32 = y.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_add() {
        let a = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let b = Tensor::new(vec![2, 2], vec![0.5, 1.5, 2.5, 3.5]);
        let y = add(&a, &b).unwrap();
        assert_eq!(y.data, vec![1.5, 3.5, 5.5, 7.5]);

        let c = Tensor::new(vec![1], vec![10.0]);
        let y2 = add(&a, &c).unwrap();
        assert_eq!(y2.data, vec![11.0, 12.0, 13.0, 14.0]);
    }

    #[test]
    fn test_extreme_fraud_fuzzing() {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let bytes = std::fs::read(format!("{}/../web/public/models/fraud_detector.onnx", manifest_dir))
            .expect("Failed to read fraud_detector.onnx for extreme testing");

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

            let score = run_fraud_model(
                &bytes,
                amount,
                old_balance_orig,
                new_balance_orig,
                old_balance_dest,
                new_balance_dest,
                is_cash_out,
                is_transfer,
            );

            assert!(score.is_ok(), "Fuzz iteration {} failed: {:?}", i, score.err());
            let val = score.unwrap();
            assert!(val >= 0.0 && val <= 1.0, "Fuzz iteration {} returned invalid score: {}", i, val);
        }
    }
}
