use super::*;

pub(super) fn register(env: &mut Env) {
    env.set_builtin("tanh", builtin_tanh);
}

fn builtin_tanh(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    unary_op(args, f32::tanh)
}
