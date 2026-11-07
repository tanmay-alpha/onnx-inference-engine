/**
 * Minimal in-browser ONNX (protobuf) metadata reader.
 *
 * Reads only the structural fields needed to describe a model (IR version,
 * producer, opset, nodes, graph inputs/outputs, initializer shapes). It never
 * decodes tensor payloads, so it is cheap even for large models. It does not
 * execute anything — inference is done by the WASM runtime or the native
 * server engine.
 */

export interface OnnxValueInfo {
  name: string;
  elemType: number | null;
  /** -1 marks a symbolic / unknown dimension; null means no shape recorded. */
  shape: number[] | null;
}

export interface OnnxNodeInfo {
  opType: string;
  name: string;
  inputs: string[];
  outputs: string[];
  attributes: string[];
}

export interface OnnxModelMeta {
  irVersion: number | null;
  producer: string;
  opset: number | null;
  nodes: OnnxNodeInfo[];
  inputs: OnnxValueInfo[];
  outputs: OnnxValueInfo[];
  initializers: { name: string; dims: number[] }[];
  opsUsed: string[];
}

export const ELEM_TYPE_NAMES: Record<number, string> = {
  1: "float32",
  2: "uint8",
  3: "int8",
  5: "int16",
  6: "int32",
  7: "int64",
  9: "bool",
  10: "float16",
  11: "float64",
  16: "bfloat16",
};

class Reader {
  pos: number;
  constructor(
    private buf: Uint8Array,
    start = 0,
    private end = buf.length,
  ) {
    this.pos = start;
  }
  eof() {
    return this.pos >= this.end;
  }
  varint(): number {
    let result = 0;
    let mul = 1;
    for (let i = 0; i < 10; i++) {
      if (this.pos >= this.end) throw new Error("Truncated varint in ONNX file");
      const b = this.buf[this.pos++];
      result += (b & 0x7f) * mul;
      if ((b & 0x80) === 0) return result;
      mul *= 128;
    }
    throw new Error("Malformed varint in ONNX file");
  }
  /** Returns [start, end) of a length-delimited field. */
  bytes(): [number, number] {
    const len = this.varint();
    const s = this.pos;
    const e = s + len;
    if (e > this.end) throw new Error("Truncated field in ONNX file");
    this.pos = e;
    return [s, e];
  }
  skip(wire: number) {
    if (wire === 0) this.varint();
    else if (wire === 1) this.pos += 8;
    else if (wire === 2) this.bytes();
    else if (wire === 5) this.pos += 4;
    else throw new Error(`Unsupported protobuf wire type ${wire}`);
  }
  sub([s, e]: [number, number]) {
    return new Reader(this.buf, s, e);
  }
  str([s, e]: [number, number]) {
    return new TextDecoder().decode(this.buf.subarray(s, e));
  }
  /** Iterate over fields: callback gets (fieldNumber, wireType). Callback must consume the value. */
  each(fn: (field: number, wire: number) => boolean | void) {
    while (!this.eof()) {
      const key = this.varint();
      const field = Math.floor(key / 8);
      const wire = key & 7;
      if (fn(field, wire) !== true) this.skip(wire);
    }
  }
}

function signedInt(v: number): number {
  // int64 negatives are encoded as 10-byte varints ≥ 2^63.
  return v >= 2 ** 63 ? v - 2 ** 64 : v;
}

function readValueInfo(r: Reader): OnnxValueInfo {
  const info: OnnxValueInfo = { name: "", elemType: null, shape: null };
  r.each((f, w) => {
    if (f === 1 && w === 2) {
      info.name = r.str(r.bytes());
      return true;
    }
    if (f === 2 && w === 2) {
      // TypeProto
      const typeR = r.sub(r.bytes());
      typeR.each((tf, tw) => {
        if (tf !== 1 || tw !== 2) return;
        const tr = typeR.sub(typeR.bytes());
        // TypeProto.Tensor
        tr.each((ff, fw) => {
          if (ff === 1 && fw === 0) {
            info.elemType = tr.varint();
            return true;
          }
          if (ff === 2 && fw === 2) {
            const shape: number[] = [];
            const sr = tr.sub(tr.bytes());
            sr.each((df, dw) => {
              if (df !== 1 || dw !== 2) return;
              let dim = -1;
              const dr = sr.sub(sr.bytes());
              dr.each((vf, vw) => {
                if (vf === 1 && vw === 0) {
                  dim = signedInt(dr.varint());
                  return true;
                }
              });
              shape.push(dim);
              return true;
            });
            info.shape = shape;
            return true;
          }
        });
        return true;
      });
      return true;
    }
  });
  return info;
}

function readNode(r: Reader): OnnxNodeInfo {
  const n: OnnxNodeInfo = { opType: "", name: "", inputs: [], outputs: [], attributes: [] };
  r.each((f, w) => {
    if (w !== 2) return;
    if (f === 1) n.inputs.push(r.str(r.bytes()));
    else if (f === 2) n.outputs.push(r.str(r.bytes()));
    else if (f === 3) n.name = r.str(r.bytes());
    else if (f === 4) n.opType = r.str(r.bytes());
    else if (f === 5) {
      const ar = r.sub(r.bytes());
      ar.each((af, aw) => {
        if (af === 1 && aw === 2) {
          n.attributes.push(ar.str(ar.bytes()));
          return true;
        }
      });
    } else return;
    return true;
  });
  return n;
}

function readInitializer(r: Reader): { name: string; dims: number[] } {
  const t = { name: "", dims: [] as number[] };
  r.each((f, w) => {
    if (f === 1 && w === 0) {
      t.dims.push(signedInt(r.varint()));
      return true;
    }
    if (f === 1 && w === 2) {
      const pr = r.sub(r.bytes());
      while (!pr.eof()) t.dims.push(signedInt(pr.varint()));
      return true;
    }
    if (f === 8 && w === 2) {
      t.name = r.str(r.bytes());
      return true;
    }
  });
  return t;
}

export function parseOnnxModel(bytes: Uint8Array): OnnxModelMeta {
  const meta: OnnxModelMeta = {
    irVersion: null,
    producer: "",
    opset: null,
    nodes: [],
    inputs: [],
    outputs: [],
    initializers: [],
    opsUsed: [],
  };
  let sawGraph = false;
  const r = new Reader(bytes);
  r.each((f, w) => {
    if (f === 1 && w === 0) {
      meta.irVersion = r.varint();
      return true;
    }
    if (f === 2 && w === 2) {
      meta.producer = r.str(r.bytes());
      return true;
    }
    if (f === 3 && w === 2) {
      const v = r.str(r.bytes());
      meta.producer = meta.producer ? `${meta.producer} ${v}` : v;
      return true;
    }
    if (f === 8 && w === 2) {
      let domain = "";
      let version: number | null = null;
      const or = r.sub(r.bytes());
      or.each((of, ow) => {
        if (of === 1 && ow === 2) {
          domain = or.str(or.bytes());
          return true;
        }
        if (of === 2 && ow === 0) {
          version = or.varint();
          return true;
        }
      });
      if ((domain === "" || domain === "ai.onnx") && version !== null) meta.opset = version;
      return true;
    }
    if (f === 7 && w === 2) {
      sawGraph = true;
      const gr = r.sub(r.bytes());
      gr.each((gf, gw) => {
        if (gw !== 2) return;
        if (gf === 1) meta.nodes.push(readNode(gr.sub(gr.bytes())));
        else if (gf === 5) meta.initializers.push(readInitializer(gr.sub(gr.bytes())));
        else if (gf === 11) meta.inputs.push(readValueInfo(gr.sub(gr.bytes())));
        else if (gf === 12) meta.outputs.push(readValueInfo(gr.sub(gr.bytes())));
        else return;
        return true;
      });
      return true;
    }
  });
  if (!sawGraph) throw new Error("Not an ONNX model: no graph found");
  const initNames = new Set(meta.initializers.map((i) => i.name));
  meta.inputs = meta.inputs.filter((i) => !initNames.has(i.name));
  meta.opsUsed = Array.from(new Set(meta.nodes.map((n) => n.opType))).sort();
  return meta;
}
