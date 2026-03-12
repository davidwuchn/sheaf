// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Minimal Python pickle parser (protocol 2-5).
//! Handles numpy/JAX arrays for ML weight loading.

use crate::core::error::SheafError;
use crate::interpreter::env::runtime_error;
use crate::interpreter::value::{Dtype, Value};
use ndarray::{ArrayD, IxDyn};
use std::collections::BTreeMap;
use std::sync::Arc;

pub fn load_pickle_bytes(data: &[u8]) -> Result<Value, SheafError> {
    let mut vm = PickleVM::new(data);
    let pv = vm.run()?;
    pickle_to_value(pv)
}

#[derive(Debug, Clone)]
enum PV {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Bytes(Vec<u8>),
    List(Vec<PV>),
    Tuple(Vec<PV>),
    Dict(Vec<(PV, PV)>),
    Global { module: String, name: String },
    NumpyArray { dtype: NpDtype, shape: Vec<usize>, data: Vec<u8> },
    NumpyDtype(NpDtype),
    Placeholder,
}

#[derive(Debug, Clone, Copy)]
enum NpDtype {
    F32,
    F64,
    I32,
    I64,
    U8,
    F16,
    BF16,
}

impl NpDtype {
    #[allow(dead_code)]
    fn byte_size(self) -> usize {
        match self {
            NpDtype::F64 | NpDtype::I64 => 8,
            NpDtype::F32 | NpDtype::I32 => 4,
            NpDtype::F16 | NpDtype::BF16 => 2,
            NpDtype::U8 => 1,
        }
    }

    fn from_str(s: &str) -> Option<NpDtype> {
        let s = s.trim_start_matches(|c: char| c == '<' || c == '>' || c == '=' || c == '|');
        match s {
            "f4" | "float32" => Some(NpDtype::F32),
            "f8" | "float64" => Some(NpDtype::F64),
            "i4" | "int32" => Some(NpDtype::I32),
            "i8" | "int64" => Some(NpDtype::I64),
            "u1" | "uint8" => Some(NpDtype::U8),
            "f2" | "float16" => Some(NpDtype::F16),
            "V2" => Some(NpDtype::BF16), // bfloat16 often stored as void2
            _ => None,
        }
    }
}

const MARK: u8 = b'(';
const STOP: u8 = b'.';
const PROTO: u8 = 0x80;
const FRAME: u8 = 0x95;
const EMPTY_DICT: u8 = b'}';
const EMPTY_LIST: u8 = b']';
const EMPTY_TUPLE: u8 = b')';
const SETITEM: u8 = b's';
const SETITEMS: u8 = b'u';
const APPEND: u8 = b'a';
const APPENDS: u8 = b'e';
const BININT1: u8 = b'K';
const BININT2: u8 = b'M';
const BININT: u8 = b'J';
const LONG1: u8 = 0x8A;
const BINFLOAT: u8 = b'G';
const NONE: u8 = b'N';
const NEWTRUE: u8 = 0x88;
const NEWFALSE: u8 = 0x89;
const SHORT_BINUNICODE: u8 = 0x8C;
const BINUNICODE: u8 = b'X';
const SHORT_BINBYTES: u8 = b'C';
const BINBYTES: u8 = b'B';
const BINBYTES8: u8 = 0x8E;
const TUPLE: u8 = b't';
const TUPLE1: u8 = 0x85;
const TUPLE2: u8 = 0x86;
const TUPLE3: u8 = 0x87;
const GLOBAL: u8 = b'c';
const STACK_GLOBAL: u8 = 0x93;
const REDUCE: u8 = b'R';
const BUILD: u8 = b'b';
const BINPUT: u8 = b'q';
const BINGET: u8 = b'h';
const LONG_BINPUT: u8 = b'r';
const LONG_BINGET: u8 = b'j';
const MEMOIZE: u8 = 0x94;
const SHORT_BINSTRING: u8 = b'U';

struct PickleVM<'a> {
    data: &'a [u8],
    pos: usize,
    stack: Vec<PV>,
    mark_stack: Vec<usize>,
    memo: BTreeMap<u32, PV>,
    memo_counter: u32,
}

impl<'a> PickleVM<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            stack: Vec::with_capacity(256),
            mark_stack: Vec::with_capacity(16),
            memo: BTreeMap::new(),
            memo_counter: 0,
        }
    }

    fn read_u8(&mut self) -> Result<u8, SheafError> {
        if self.pos >= self.data.len() {
            return Err(runtime_error("pickle: unexpected EOF"));
        }
        let b = self.data[self.pos];
        self.pos += 1;
        Ok(b)
    }

    fn read_u16_le(&mut self) -> Result<u16, SheafError> {
        if self.pos + 2 > self.data.len() {
            return Err(runtime_error("pickle: unexpected EOF"));
        }
        let v = u16::from_le_bytes([self.data[self.pos], self.data[self.pos + 1]]);
        self.pos += 2;
        Ok(v)
    }

    fn read_i32_le(&mut self) -> Result<i32, SheafError> {
        if self.pos + 4 > self.data.len() {
            return Err(runtime_error("pickle: unexpected EOF"));
        }
        let v = i32::from_le_bytes([
            self.data[self.pos], self.data[self.pos + 1],
            self.data[self.pos + 2], self.data[self.pos + 3],
        ]);
        self.pos += 4;
        Ok(v)
    }

    fn read_u32_le(&mut self) -> Result<u32, SheafError> {
        if self.pos + 4 > self.data.len() {
            return Err(runtime_error("pickle: unexpected EOF"));
        }
        let v = u32::from_le_bytes([
            self.data[self.pos], self.data[self.pos + 1],
            self.data[self.pos + 2], self.data[self.pos + 3],
        ]);
        self.pos += 4;
        Ok(v)
    }

    fn read_u64_le(&mut self) -> Result<u64, SheafError> {
        if self.pos + 8 > self.data.len() {
            return Err(runtime_error("pickle: unexpected EOF"));
        }
        let v = u64::from_le_bytes([
            self.data[self.pos], self.data[self.pos + 1],
            self.data[self.pos + 2], self.data[self.pos + 3],
            self.data[self.pos + 4], self.data[self.pos + 5],
            self.data[self.pos + 6], self.data[self.pos + 7],
        ]);
        self.pos += 8;
        Ok(v)
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], SheafError> {
        if self.pos + n > self.data.len() {
            return Err(runtime_error("pickle: unexpected EOF"));
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn read_line(&mut self) -> Result<&'a [u8], SheafError> {
        let start = self.pos;
        while self.pos < self.data.len() && self.data[self.pos] != b'\n' {
            self.pos += 1;
        }
        if self.pos >= self.data.len() {
            return Err(runtime_error("pickle: unexpected EOF reading line"));
        }
        let line = &self.data[start..self.pos];
        self.pos += 1; // skip \n
        Ok(line)
    }

    fn pop(&mut self) -> Result<PV, SheafError> {
        self.stack.pop().ok_or_else(|| runtime_error("pickle: stack underflow"))
    }

    fn pop_mark(&mut self) -> Result<Vec<PV>, SheafError> {
        let mark_pos = self.mark_stack.pop()
            .ok_or_else(|| runtime_error("pickle: MARK stack underflow"))?;
        let items = self.stack.split_off(mark_pos);
        Ok(items)
    }

    fn run(&mut self) -> Result<PV, SheafError> {
        loop {
            if self.pos >= self.data.len() {
                return Err(runtime_error("pickle: unexpected EOF"));
            }
            let opcode = self.data[self.pos];
            self.pos += 1;

            match opcode {
                PROTO => {
                    let version = self.read_u8()?;
                    if version > 5 {
                        return Err(runtime_error(format!(
                            "pickle: unsupported protocol version {}", version
                        )));
                    }
                }

                FRAME => {
                    // Advisory frame length, skip 8 bytes
                    self.pos += 8;
                }

                STOP => {
                    return self.pop();
                }

                MARK => {
                    self.mark_stack.push(self.stack.len());
                }

                EMPTY_DICT => self.stack.push(PV::Dict(Vec::new())),
                EMPTY_LIST => self.stack.push(PV::List(Vec::new())),
                EMPTY_TUPLE => self.stack.push(PV::Tuple(Vec::new())),

                NONE => self.stack.push(PV::None),
                NEWTRUE => self.stack.push(PV::Bool(true)),
                NEWFALSE => self.stack.push(PV::Bool(false)),

                BININT1 => {
                    let v = self.read_u8()? as i64;
                    self.stack.push(PV::Int(v));
                }

                BININT2 => {
                    let v = self.read_u16_le()? as i64;
                    self.stack.push(PV::Int(v));
                }

                BININT => {
                    let v = self.read_i32_le()? as i64;
                    self.stack.push(PV::Int(v));
                }

                LONG1 => {
                    let n = self.read_u8()? as usize;
                    let bytes = self.read_bytes(n)?;
                    // Little-endian signed integer
                    let mut val: i64 = 0;
                    for (i, &b) in bytes.iter().enumerate() {
                        val |= (b as i64) << (i * 8);
                    }
                    // Sign extension
                    if n > 0 && bytes[n - 1] & 0x80 != 0 {
                        for i in n..8 {
                            val |= 0xFFi64 << (i * 8);
                        }
                    }
                    self.stack.push(PV::Int(val));
                }

                BINFLOAT => {
                    let bytes = self.read_bytes(8)?;
                    let v = f64::from_be_bytes([
                        bytes[0], bytes[1], bytes[2], bytes[3],
                        bytes[4], bytes[5], bytes[6], bytes[7],
                    ]);
                    self.stack.push(PV::Float(v));
                }

                SHORT_BINUNICODE => {
                    let n = self.read_u8()? as usize;
                    let bytes = self.read_bytes(n)?;
                    let s = std::str::from_utf8(bytes)
                        .map_err(|e| runtime_error(format!("pickle: invalid UTF-8: {}", e)))?;
                    self.stack.push(PV::Str(s.to_string()));
                }

                BINUNICODE => {
                    let n = self.read_u32_le()? as usize;
                    let bytes = self.read_bytes(n)?;
                    let s = std::str::from_utf8(bytes)
                        .map_err(|e| runtime_error(format!("pickle: invalid UTF-8: {}", e)))?;
                    self.stack.push(PV::Str(s.to_string()));
                }

                SHORT_BINSTRING => {
                    let n = self.read_u8()? as usize;
                    let bytes = self.read_bytes(n)?;
                    let s = std::str::from_utf8(bytes).unwrap_or("").to_string();
                    self.stack.push(PV::Str(s));
                }

                SHORT_BINBYTES => {
                    let n = self.read_u8()? as usize;
                    let bytes = self.read_bytes(n)?;
                    self.stack.push(PV::Bytes(bytes.to_vec()));
                }

                BINBYTES => {
                    let n = self.read_u32_le()? as usize;
                    let bytes = self.read_bytes(n)?;
                    self.stack.push(PV::Bytes(bytes.to_vec()));
                }

                BINBYTES8 => {
                    let n = self.read_u64_le()? as usize;
                    let bytes = self.read_bytes(n)?;
                    self.stack.push(PV::Bytes(bytes.to_vec()));
                }

                SETITEM => {
                    let val = self.pop()?;
                    let key = self.pop()?;
                    if let Some(PV::Dict(pairs)) = self.stack.last_mut() {
                        pairs.push((key, val));
                    } else {
                        return Err(runtime_error("pickle: SETITEM on non-dict"));
                    }
                }

                SETITEMS => {
                    let items = self.pop_mark()?;
                    if let Some(PV::Dict(pairs)) = self.stack.last_mut() {
                        for chunk in items.chunks(2) {
                            if chunk.len() == 2 {
                                pairs.push((chunk[0].clone(), chunk[1].clone()));
                            }
                        }
                    } else {
                        return Err(runtime_error("pickle: SETITEMS on non-dict"));
                    }
                }

                APPEND => {
                    let val = self.pop()?;
                    if let Some(PV::List(items)) = self.stack.last_mut() {
                        items.push(val);
                    } else {
                        return Err(runtime_error("pickle: APPEND on non-list"));
                    }
                }

                APPENDS => {
                    let items = self.pop_mark()?;
                    if let Some(PV::List(list)) = self.stack.last_mut() {
                        list.extend(items);
                    } else {
                        return Err(runtime_error("pickle: APPENDS on non-list"));
                    }
                }

                TUPLE => {
                    let items = self.pop_mark()?;
                    self.stack.push(PV::Tuple(items));
                }

                TUPLE1 => {
                    let a = self.pop()?;
                    self.stack.push(PV::Tuple(vec![a]));
                }

                TUPLE2 => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(PV::Tuple(vec![a, b]));
                }

                TUPLE3 => {
                    let c = self.pop()?;
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(PV::Tuple(vec![a, b, c]));
                }

                GLOBAL => {
                    let module_line = self.read_line()?;
                    let name_line = self.read_line()?;
                    let module = std::str::from_utf8(module_line).unwrap_or("").to_string();
                    let name = std::str::from_utf8(name_line).unwrap_or("").to_string();
                    self.stack.push(PV::Global { module, name });
                }

                STACK_GLOBAL => {
                    let name = self.pop()?;
                    let module = self.pop()?;
                    let (m, n) = match (&module, &name) {
                        (PV::Str(m), PV::Str(n)) => (m.clone(), n.clone()),
                        _ => return Err(runtime_error("pickle: STACK_GLOBAL expects strings")),
                    };
                    self.stack.push(PV::Global { module: m, name: n });
                }

                REDUCE => {
                    let args = self.pop()?;
                    let callable = self.pop()?;
                    let result = self.handle_reduce(callable, args)?;
                    self.stack.push(result);
                }

                BUILD => {
                    let state = self.pop()?;
                    let obj = self.pop()?;
                    let result = self.handle_build(obj, state)?;
                    self.stack.push(result);
                }

                BINPUT => {
                    let idx = self.read_u8()? as u32;
                    if let Some(top) = self.stack.last() {
                        self.memo.insert(idx, top.clone());
                    }
                }

                LONG_BINPUT => {
                    let idx = self.read_u32_le()?;
                    if let Some(top) = self.stack.last() {
                        self.memo.insert(idx, top.clone());
                    }
                }

                BINGET => {
                    let idx = self.read_u8()? as u32;
                    let val = self.memo.get(&idx)
                        .ok_or_else(|| runtime_error(format!("pickle: memo {} not found", idx)))?
                        .clone();
                    self.stack.push(val);
                }

                LONG_BINGET => {
                    let idx = self.read_u32_le()?;
                    let val = self.memo.get(&idx)
                        .ok_or_else(|| runtime_error(format!("pickle: memo {} not found", idx)))?
                        .clone();
                    self.stack.push(val);
                }

                MEMOIZE => {
                    let idx = self.memo_counter;
                    self.memo_counter += 1;
                    if let Some(top) = self.stack.last() {
                        self.memo.insert(idx, top.clone());
                    }
                }

                other => {
                    return Err(runtime_error(format!(
                        "pickle: unsupported opcode 0x{:02X} ('{}') at offset {}",
                        other, other as char, self.pos - 1
                    )));
                }
            }
        }
    }

    fn handle_reduce(&self, callable: PV, args: PV) -> Result<PV, SheafError> {
        let (module, name) = match &callable {
            PV::Global { module, name } => (module.as_str(), name.as_str()),
            _ => return Ok(PV::Placeholder),
        };

        match (module, name) {
            // numpy._core.multiarray._reconstruct or numpy.core.multiarray._reconstruct
            (m, "_reconstruct") if m.contains("multiarray") => {
                // Args: (subtype, shape_tuple, dtype_char)
                // Returns placeholder; BUILD will fill in the actual data
                Ok(PV::Placeholder)
            }

            // numpy.dtype
            ("numpy", "dtype") => {
                let args_vec = match args {
                    PV::Tuple(v) => v,
                    _ => return Ok(PV::Placeholder),
                };
                if let Some(PV::Str(s)) = args_vec.first() {
                    if let Some(dt) = NpDtype::from_str(s) {
                        return Ok(PV::NumpyDtype(dt));
                    }
                }
                Ok(PV::Placeholder)
            }

            // _codecs.encode, used to embed raw bytes as latin1 string
            ("_codecs", "encode") => {
                let args_vec = match args {
                    PV::Tuple(v) => v,
                    _ => return Ok(PV::Bytes(Vec::new())),
                };
                match args_vec.first() {
                    Some(PV::Str(s)) => {
                        // latin1 encode: each char maps to its byte value
                        let bytes: Vec<u8> = s.chars().map(|c| c as u8).collect();
                        Ok(PV::Bytes(bytes))
                    }
                    Some(PV::Bytes(b)) => Ok(PV::Bytes(b.clone())),
                    _ => Ok(PV::Bytes(Vec::new())),
                }
            }

            // JAX array reconstruction
            // Args: (_reconstruct_global, (ndarray, (0,), b'b'), state_tuple, extra_dict)
            // state_tuple: (version, shape, dtype, fortran_order, raw_bytes)
            (m, "_reconstruct_array") if m.contains("jax") => {
                let args_vec = match args {
                    PV::Tuple(v) => v,
                    _ => return Ok(PV::Placeholder),
                };
                // First check if there's already a NumpyArray (from a different path)
                for a in &args_vec {
                    if let PV::NumpyArray { .. } = a {
                        return Ok(a.clone());
                    }
                }
                // Extract state tuple (index 2) and reconstruct from it
                if let Some(PV::Tuple(state)) = args_vec.get(2) {
                    if let Some(arr) = extract_numpy_from_state(state) {
                        return Ok(arr);
                    }
                }
                Ok(PV::Placeholder)
            }

            // collections.OrderedDict
            ("collections", "OrderedDict") => {
                Ok(PV::Dict(Vec::new()))
            }

            // Fallback: just return the args or placeholder
            _ => Ok(PV::Placeholder),
        }
    }


    fn handle_build(&self, obj: PV, state: PV) -> Result<PV, SheafError> {
        match obj {
            // numpy array Placeholder -> fill from BUILD state
            PV::Placeholder => {
                let state_vec = match state {
                    PV::Tuple(v) => v,
                    _ => return Ok(state),
                };
                if let Some(arr) = extract_numpy_from_state(&state_vec) {
                    return Ok(arr);
                }
                Ok(PV::Tuple(state_vec))
            }

            // OrderedDict BUILD, state is list of key-value pairs
            PV::Dict(mut pairs) => {
                match state {
                    PV::Dict(more) => {
                        pairs.extend(more);
                        Ok(PV::Dict(pairs))
                    }
                    PV::List(items) => {
                        // List of (key, value) tuples
                        for item in items {
                            if let PV::Tuple(kv) = item {
                                if kv.len() == 2 {
                                    pairs.push((kv[0].clone(), kv[1].clone()));
                                }
                            }
                        }
                        Ok(PV::Dict(pairs))
                    }
                    _ => Ok(PV::Dict(pairs))
                }
            }

            // NumpyArray getting its state updated
            PV::NumpyArray { dtype, shape, data } => {
                if let PV::Tuple(state_vec) = state {
                    if let Some(arr) = extract_numpy_from_state(&state_vec) {
                        return Ok(arr);
                    }
                }
                Ok(PV::NumpyArray { dtype, shape, data })
            }

            other => Ok(other),
        }
    }
}

fn extract_numpy_from_state(state: &[PV]) -> Option<PV> {
    let mut shape: Option<Vec<usize>> = None;
    let mut np_dtype: Option<NpDtype> = None;
    let mut raw_data: Option<&[u8]> = None;

    for item in state {
        match item {
            PV::Tuple(t) => {
                let maybe_shape: Option<Vec<usize>> = t.iter().map(|v| match v {
                    PV::Int(n) => Some(*n as usize),
                    _ => None,
                }).collect();
                if let Some(s) = maybe_shape {
                    if shape.is_none() || !s.is_empty() {
                        shape = Some(s);
                    }
                }
            }
            PV::NumpyDtype(dt) => np_dtype = Some(*dt),
            PV::Bytes(b) => {
                if raw_data.is_none() || b.len() > raw_data.map_or(0, |r| r.len()) {
                    raw_data = Some(b);
                }
            }
            _ => {}
        }
    }

    if let (Some(shape), Some(dtype), Some(data)) = (shape, np_dtype, raw_data) {
        Some(PV::NumpyArray { dtype, shape, data: data.to_vec() })
    } else {
        None
    }
}

fn numpy_bytes_to_f32(data: &[u8], dtype: NpDtype) -> Vec<f32> {
    match dtype {
        NpDtype::F32 => {
            data.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }
        NpDtype::F64 => {
            data.chunks_exact(8)
                .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
                .collect()
        }
        NpDtype::I32 => {
            data.chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
                .collect()
        }
        NpDtype::I64 => {
            data.chunks_exact(8)
                .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
                .collect()
        }
        NpDtype::U8 => {
            data.iter().map(|&b| b as f32).collect()
        }
        NpDtype::F16 => {
            data.chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f16_to_f32(bits)
                })
                .collect()
        }
        NpDtype::BF16 => {
            data.chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    bf16_to_f32(bits)
                })
                .collect()
        }
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let frac = (h & 0x3FF) as u32;
    if exp == 0 {
        if frac == 0 {
            f32::from_bits(sign << 31)
        } else {
            let mut e = 0u32;
            let mut f = frac;
            while f & 0x400 == 0 { f <<= 1; e += 1; }
            f &= 0x3FF;
            let e = 127 - 15 + 1 - e;
            f32::from_bits((sign << 31) | (e << 23) | (f << 13))
        }
    } else if exp == 31 {
        if frac == 0 {
            f32::from_bits((sign << 31) | (0xFF << 23))
        } else {
            f32::NAN
        }
    } else {
        let e = exp + 127 - 15;
        f32::from_bits((sign << 31) | (e << 23) | (frac << 13))
    }
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn pickle_to_value(pv: PV) -> Result<Value, SheafError> {
    match pv {
        PV::None => Ok(Value::Nil),
        PV::Bool(b) => Ok(Value::Bool(b)),
        PV::Int(n) => Ok(Value::Int(n)),
        PV::Float(f) => Ok(Value::Float(f as f32)),
        PV::Str(s) => Ok(Value::String(s)),
        PV::Bytes(b) => {
            // Expose raw bytes as a list of ints
            Ok(Value::List(b.into_iter().map(|x| Value::Int(x as i64)).collect()))
        }
        PV::List(items) => {
            let vals: Result<Vec<Value>, _> = items.into_iter().map(pickle_to_value).collect();
            Ok(Value::List(vals?))
        }
        PV::Tuple(items) => {
            let vals: Result<Vec<Value>, _> = items.into_iter().map(pickle_to_value).collect();
            Ok(Value::Tuple(vals?))
        }
        PV::Dict(pairs) => {
            let mut map = BTreeMap::new();
            for (k, v) in pairs {
                let key = match k {
                    PV::Str(s) => s,
                    PV::Int(n) => n.to_string(),
                    _ => format!("{:?}", k),
                };
                map.insert(key, pickle_to_value(v)?);
            }
            Ok(Value::Dict(map))
        }
        PV::NumpyArray { dtype, shape, data } => {
            let values = numpy_bytes_to_f32(&data, dtype);
            let expected: usize = shape.iter().product();
            if values.len() != expected {
                return Err(runtime_error(format!(
                    "pickle: numpy array shape {:?} expects {} elements, got {}",
                    shape, expected, values.len()
                )));
            }
            let sheaf_dtype = match dtype {
                NpDtype::I32 | NpDtype::I64 => Dtype::I32,
                _ => Dtype::F32,
            };
            let arr = ArrayD::from_shape_vec(IxDyn(&shape), values)
                .map_err(|e| runtime_error(format!("pickle: array reshape: {}", e)))?;
            Ok(Value::Tensor { data: Arc::new(arr), dtype: sheaf_dtype })
        }
        PV::NumpyDtype(_) | PV::Global { .. } | PV::Placeholder => {
            Ok(Value::Nil)
        }
    }
}
