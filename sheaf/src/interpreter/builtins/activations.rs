use super::*;

pub(super) fn register(env: &mut Env) {
    register_native_builtin(env, OpId::Tanh, builtin_tanh);
}

fn builtin_tanh(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    unary_op(args, f32::tanh)
}
