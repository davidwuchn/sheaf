// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Integration tests for the Sheaf interpreter.

use sheaf_compiler::interpreter::eval_exprs;

fn eval(src: &str) -> String {
    match eval_exprs(src) {
        Ok(val) => format!("{}", val),
        Err(e) => panic!("Eval error: {}", e),
    }
}

#[test]
fn test_add_integers() {
    assert_eq!(eval("(+ 1 2)"), "3");
}

#[test]
fn test_add_vectors() {
    assert_eq!(eval("(+ [1 2] [3 4])"), "[4. 6.]");
}

#[test]
fn test_add_scalar_vector() {
    assert_eq!(eval("(+ 1 [2 3] 4)"), "[7. 8.]");
}

#[test]
fn test_sub_unary() {
    assert_eq!(eval("(- 5)"), "-5");
}

#[test]
fn test_sub_binary() {
    assert_eq!(eval("(- 10 3)"), "7");
}

#[test]
fn test_sub_vectors() {
    assert_eq!(eval("(- [5 10] [2 3])"), "[3. 7.]");
}

#[test]
fn test_mul_integers() {
    assert_eq!(eval("(* 2 3)"), "6");
}

#[test]
fn test_mul_scalar_vector() {
    assert_eq!(eval("(* [1 2] 3)"), "[3. 6.]");
}

#[test]
fn test_mul_vectors() {
    assert_eq!(eval("(* [2 3] [4 5])"), "[ 8. 15.]");
}

#[test]
fn test_div() {
    assert_eq!(eval("(/ 10.0 2.0)"), "5.0");
}

#[test]
fn test_div_vector() {
    assert_eq!(eval("(/ [10 20] 2)"), "[ 5. 10.]");
}

#[test]
fn test_floor_div() {
    assert_eq!(eval("(// 7 2)"), "3");
}

#[test]
fn test_floor_div_vector() {
    assert_eq!(eval("(// [7 9] 2)"), "[3. 4.]");
}

#[test]
fn test_mod() {
    assert_eq!(eval("(mod 7 3)"), "1");
}

#[test]
fn test_mod_alias() {
    assert_eq!(eval("(% 7 3)"), "1");
}

#[test]
fn test_mod_vector() {
    assert_eq!(eval("(mod [7 9] 2)"), "[1. 1.]");
}

#[test]
fn test_mod_negative() {
    // Python-style: mod(-7, 3) = 2
    assert_eq!(eval("(mod -7 3)"), "2");
}

#[test]
fn test_pow() {
    assert_eq!(eval("(** 2.0 3.0)"), "8.0");
}

#[test]
fn test_pow_vector() {
    assert_eq!(eval("(** [2 3] 2)"), "[4. 9.]");
}

#[test]
fn test_abs() {
    assert_eq!(eval("(abs -5.0)"), "5.0");
}

#[test]
fn test_abs_vector() {
    assert_eq!(eval("(abs [-3 2 -1])"), "[3. 2. 1.]");
}

#[test]
fn test_exp_zero() {
    assert_eq!(eval("(exp 0.0)"), "1.0");
}

#[test]
fn test_log_one() {
    assert_eq!(eval("(log 1.0)"), "0.0");
}

#[test]
fn test_sqrt() {
    assert_eq!(eval("(sqrt 4.0)"), "2.0");
}

#[test]
fn test_sqrt_vector() {
    assert_eq!(eval("(sqrt [1.0 4.0 9.0])"), "[1. 2. 3.]");
}

#[test]
fn test_matmul_dot() {
    assert_eq!(eval("(@ [1 2 3] [4 5 6])"), "32.0");
}

#[test]
fn test_matmul_2d() {
    assert_eq!(eval("(@ [[1 2 3] [4 5 6]] [[1 2] [3 4] [5 6]])"), "[[22. 28.]\n [49. 64.]]");
}

#[test]
fn test_eq_true() {
    assert_eq!(eval("(= 1 1)"), "True");
}

#[test]
fn test_eq_vectors_true() {
    assert_eq!(eval("(= [1 2] [1 2])"), "True");
}

#[test]
fn test_eq_vectors_false() {
    assert_eq!(eval("(= [1 2] [1 3])"), "False");
}

#[test]
fn test_gt_scalar() {
    assert_eq!(eval("(> 5 2)"), "True");
}

#[test]
fn test_not_true() {
    assert_eq!(eval("(not true)"), "False");
}

#[test]
fn test_not_false() {
    assert_eq!(eval("(not false)"), "True");
}

#[test]
fn test_and_short_circuit() {
    assert_eq!(eval("(and true false)"), "False");
}

#[test]
fn test_and_truthy() {
    assert_eq!(eval("(and 1 2 3)"), "3");
}

#[test]
fn test_or_short_circuit() {
    assert_eq!(eval("(or false true)"), "True");
}

#[test]
fn test_or_nil() {
    assert_eq!(eval("(or nil 42)"), "42");
}

#[test]
fn test_if_true() {
    assert_eq!(eval("(if (> 1 0) :yes :no)"), ":yes");
}

#[test]
fn test_if_false() {
    assert_eq!(eval("(if false :a :b)"), ":b");
}

#[test]
fn test_let() {
    assert_eq!(eval("(let [x 1] x)"), "1");
}

#[test]
fn test_fn_call() {
    assert_eq!(eval("((fn [x] (+ x 1)) 10)"), "11");
}

#[test]
fn test_defn_call() {
    assert_eq!(eval("(defn add [x y] (+ x y)) (add 3 4)"), "7");
}

#[test]
fn test_let_with_fn() {
    assert_eq!(eval("(let [f (fn [x] (* x 2))] (f 5))"), "10");
}

#[test]
fn test_neq_scalar() {
    assert_eq!(eval("(!= 1 2)"), "True");
}

#[test]
fn test_vector_2d() {
    assert_eq!(eval("[[1 2] [3 4]]"), "[[1. 2.]
 [3. 4.]]");
}

#[test]
fn test_ndim() {
    assert_eq!(eval("(ndim [1 2 3])"), "1");
}

#[test]
fn test_len() {
    assert_eq!(eval("(len [1 2 3])"), "3");
}

#[test]
fn test_shape() {
    // shape of [1 2 3] → [3]
    let result = eval("(shape [1 2 3])");
    assert_eq!(result, "[3.]");
}

#[test]
fn test_int_from_float() {
    assert_eq!(eval("(int 3.7)"), "3");
}

#[test]
fn test_float_from_int() {
    assert_eq!(eval("(float 42)"), "42.0");
}

#[test]
fn test_nested_let() {
    assert_eq!(eval("(let [x 10 y (+ x 5)] y)"), "15");
}

#[test]
fn test_do_block() {
    assert_eq!(eval("(do 1 2 3)"), "3");
}

#[test]
fn test_quote_list() {
    assert_eq!(eval("'[1 2 3]"), "[1, 2, 3]");
}

// Phase 2: Activations

#[test]
fn test_relu_negative() {
    assert_eq!(eval("(relu -1.0)"), "0.0");
}

#[test]
fn test_relu_vector() {
    assert_eq!(eval("(relu [1.0 -2.0 3.0])"), "[1. 0. 3.]");
}

#[test]
fn test_leaky_relu_negative() {
    assert_eq!(eval("(leaky-relu -1.0)"), "-0.009999999776482582");
}

#[test]
fn test_sigmoid_zero() {
    assert_eq!(eval("(sigmoid 0.0)"), "0.5");
}

#[test]
fn test_sigmoid_large_negative() {
    assert_eq!(eval("(sigmoid -100.0)"), "0.0");
}

#[test]
fn test_tanh_zero() {
    assert_eq!(eval("(tanh 0.0)"), "0.0");
}

#[test]
fn test_silu_zero() {
    assert_eq!(eval("(silu 0.0)"), "0.0");
}

#[test]
fn test_softmax_uniform() {
    assert_eq!(eval("(softmax [1.0 1.0])"), "[0.5 0.5]");
}

// Phase 2: Reductions

#[test]
fn test_sum_1d() {
    assert_eq!(eval("(sum [1 2 3])"), "6.0");
}

#[test]
fn test_sum_axis0() {
    assert_eq!(eval("(sum [[1 2] [3 4]] :axis 0)"), "[4. 6.]");
}

#[test]
fn test_mean_1d() {
    assert_eq!(eval("(mean [1.0 2.0 3.0])"), "2.0");
}

#[test]
fn test_mean_axis1() {
    assert_eq!(eval("(mean [[1 2] [3 4]] :axis 1)"), "[1.5 3.5]");
}

#[test]
fn test_product_1d() {
    assert_eq!(eval("(product [1 2 3])"), "6.0");
}

#[test]
fn test_product_axis() {
    assert_eq!(eval("(product [2.0 3.0 4.0] :axis 0)"), "24.0");
}

#[test]
fn test_min_1d() {
    assert_eq!(eval("(min [3 1 4])"), "1.0");
}

#[test]
fn test_max_1d() {
    assert_eq!(eval("(max [1.0 2.0 10.0])"), "10.0");
}

#[test]
fn test_minimum_elementwise() {
    assert_eq!(eval("(minimum [1 10] [5 2])"), "[1. 2.]");
}

#[test]
fn test_maximum_elementwise() {
    assert_eq!(eval("(maximum [1 10] [5 2])"), "[ 5. 10.]");
}

#[test]
fn test_argmax_1d() {
    assert_eq!(eval("(argmax [3 1 4 1 5])"), "4");
}

#[test]
fn test_argmax_axis1() {
    assert_eq!(eval("(argmax [[1 5] [3 2]] :axis 1)"), "[1 0]");
}

#[test]
fn test_argmin_1d() {
    assert_eq!(eval("(argmin [3 1 4 1 5])"), "1");
}

#[test]
fn test_argmin_axis0() {
    assert_eq!(eval("(argmin [[1 5] [3 2]] :axis 0)"), "[0 1]");
}

// Phase 2: Tensor construction

#[test]
fn test_zeros() {
    assert_eq!(eval("(zeros '[3])"), "[0. 0. 0.]");
}

#[test]
fn test_ones_2d() {
    assert_eq!(eval("(ones '[2 3])"), "[[1. 1. 1.]\n [1. 1. 1.]]");
}

#[test]
fn test_arange_1arg() {
    assert_eq!(eval("(arange 5)"), "[0 1 2 3 4]");
}

#[test]
fn test_arange_2arg() {
    assert_eq!(eval("(arange 2 7)"), "[2 3 4 5 6]");
}

#[test]
fn test_arange_3arg() {
    assert_eq!(eval("(arange 0 10 2)"), "[0 2 4 6 8]");
}

#[test]
fn test_eye_square() {
    assert_eq!(eval("(eye 3)"), "[[1. 0. 0.]\n [0. 1. 0.]\n [0. 0. 1.]]");
}

#[test]
fn test_eye_rect() {
    assert_eq!(eval("(eye 2 4)"), "[[1. 0. 0. 0.]\n [0. 1. 0. 0.]]");
}

#[test]
fn test_one_hot_scalar() {
    assert_eq!(eval("(one-hot 1 3)"), "[0. 1. 0.]");
}

#[test]
fn test_tril() {
    assert_eq!(eval("(tril [[1 2] [3 4]])"), "[[1. 0.]\n [3. 4.]]");
}

// Phase 2: Tensor ops

#[test]
fn test_reshape() {
    assert_eq!(eval("(reshape (arange 6) '[2 3])"), "[[0 1 2]\n [3 4 5]]");
}

#[test]
fn test_reshape_infer() {
    assert_eq!(eval("(reshape (arange 9) '[3 -1])"), "[[0 1 2]\n [3 4 5]\n [6 7 8]]");
}

#[test]
fn test_slice() {
    assert_eq!(eval("(slice [0 1 2 3 4] 1 4)"), "[1. 2. 3.]");
}

#[test]
fn test_roll_positive() {
    assert_eq!(eval("(roll [1 2 3] 1)"), "[3. 1. 2.]");
}

#[test]
fn test_roll_negative() {
    assert_eq!(eval("(roll [1 2 3] -1)"), "[2. 3. 1.]");
}

#[test]
fn test_where_op() {
    assert_eq!(eval("(where (> [1 3 2] 2) [10 20 30] 0)"), "[ 0. 20.  0.]");
}

#[test]
fn test_index_update_scalar() {
    assert_eq!(eval("(index-update [1 2 3 4 5] 2 99)"), "[ 1.  2. 99.  4.  5.]");
}

#[test]
fn test_index_update_row() {
    assert_eq!(eval("(index-update [[1 2] [3 4]] 0 [10 20])"), "[[10. 20.]\n [ 3.  4.]]");
}

// Phase 2: List ops

#[test]
fn test_first() {
    assert_eq!(eval("(first '[1 2 3])"), "1");
}

#[test]
fn test_second() {
    assert_eq!(eval("(second '[1 2 3])"), "2");
}

#[test]
fn test_last() {
    assert_eq!(eval("(last '[1 2 3])"), "3");
}

#[test]
fn test_rest() {
    assert_eq!(eval("(rest '[1 2 3])"), "[2, 3]");
}

#[test]
fn test_nth_list() {
    assert_eq!(eval("(nth '[10 20 30] 1)"), "20");
}

#[test]
fn test_nth_tensor() {
    assert_eq!(eval("(nth [10 20 30] 1)"), "20.0");
}

#[test]
fn test_cons() {
    assert_eq!(eval("(cons 0 '[1 2 3])"), "[0, 1, 2, 3]");
}

#[test]
fn test_append_list() {
    assert_eq!(eval("(append '[1 2] 3)"), "[1, 2, 3]");
}

#[test]
fn test_empty_true() {
    assert_eq!(eval("(empty? '[])"), "True");
}

#[test]
fn test_empty_false() {
    assert_eq!(eval("(empty? '[1])"), "False");
}

#[test]
fn test_empty_tensor() {
    assert_eq!(eval("(empty? [1 2 3])"), "False");
}

#[test]
fn test_count_list() {
    assert_eq!(eval("(count '[1 2 3 4])"), "4");
}

#[test]
fn test_count_2d() {
    assert_eq!(eval("(count [[1 2] [3 4]])"), "2");
}

// Phase 2: Dict ops

#[test]
fn test_get_dict() {
    assert_eq!(eval("(get {:a 1 :b 2} :a)"), "1");
}

#[test]
fn test_get_dict_default() {
    assert_eq!(eval("(get {:a 1} :missing 99)"), "99");
}

#[test]
fn test_get_in_nested() {
    assert_eq!(eval("(get-in {:layers {:l1 {:w 10}}} [:layers :l1 :w])"), "10");
}

#[test]
fn test_get_in_default() {
    assert_eq!(eval("(get-in {:a 1} [:layers :l1 :depth] 12)"), "12");
}

#[test]
fn test_get_in_tensor() {
    assert_eq!(eval("(get-in {:l1 {:w [[1 2] [3 4]]}} [:l1 :w 0])"), "[1. 2.]");
}

#[test]
fn test_assoc() {
    assert_eq!(eval("(assoc {:a 1} :b 2)"), "{:a 1, :b 2}");
}

#[test]
fn test_assoc_overwrite() {
    assert_eq!(eval("(assoc {:a 1 :b 2} :a 10)"), "{:a 10, :b 2}");
}

#[test]
fn test_dissoc() {
    assert_eq!(eval("(dissoc {:a 1 :b 2 :c 3} [:b])"), "{:a 1, :c 3}");
}

#[test]
fn test_merge() {
    assert_eq!(eval("(merge {:a 1} {:b 2})"), "{:a 1, :b 2}");
}

#[test]
fn test_keys() {
    let result = eval("(keys {:a 1 :b 2 :c 3})");
    assert!(result.contains("a") && result.contains("b") && result.contains("c"));
}

#[test]
fn test_vals() {
    assert_eq!(eval("(vals {:a 1 :b 2 :c 3})"), "[1, 2, 3]");
}

// Phase 3: Higher-order with lambdas

#[test]
fn test_map_lambda() {
    assert_eq!(eval("(map (fn [x] (* x 2)) '(1 2 3))"), "[2, 4, 6]");
}

#[test]
fn test_map_lambda_strings() {
    assert_eq!(eval("(map (fn [x] (str x)) '(1 2 3))"), "['1', '2', '3']");
}

#[test]
fn test_filter_lambda() {
    assert_eq!(eval("(filter (fn [x] (> x 2)) '(1 2 3 4 5))"), "[3, 4, 5]");
}

#[test]
fn test_reduce_lambda() {
    assert_eq!(eval("(reduce (fn [a b] (+ a b)) 0 '(1 2 3 4))"), "10");
}

#[test]
fn test_reduce_lambda_mul() {
    assert_eq!(eval("(reduce (fn [a b] (* a b)) 1 '(1 2 3 4))"), "24");
}

#[test]
fn test_apply_lambda() {
    assert_eq!(eval("(apply (fn [a b c] (+ a b c)) '(10 20 30))"), "60");
}

#[test]
fn test_map_builtin() {
    assert_eq!(eval("(map abs '(-1 -2 3))"), "[1.0, 2.0, 3.0]");
}

#[test]
fn test_reduce_builtin() {
    assert_eq!(eval("(reduce + 0 '(1 2 3 4))"), "10");
}

#[test]
fn test_apply_builtin() {
    assert_eq!(eval("(apply + '(1 2 3))"), "6");
}

#[test]
fn test_filter_builtin() {
    // not is a builtin that returns falsy for 0/nil/false
    assert_eq!(eval("(filter not '(nil 1 nil 2))"), "[nil, nil]");
}

#[test]
fn test_map_with_defn() {
    assert_eq!(
        eval("(defn double [x] (* x 2)) (map double '(1 2 3))"),
        "[2, 4, 6]"
    );
}

#[test]
fn test_reduce_tensor() {
    assert_eq!(eval("(reduce (fn [a b] (+ a b)) 0 [1 2 3 4])"), "10.0");
}

#[test]
fn test_scan_list() {
    // scan returns (final-carry, [intermediates])
    assert_eq!(
        eval("(first (scan (fn [a b] (+ a b)) 0 '(1 2 3 4)))"),
        "10"
    );
}

#[test]
fn test_scan_outputs() {
    // intermediate carries: 1, 3, 6, 10
    assert_eq!(
        eval("(second (scan (fn [a b] (+ a b)) 0 '(1 2 3 4)))"),
        "[1, 3, 6, 10]"
    );
}

// Phase 2: String

#[test]
fn test_str() {
    assert_eq!(eval("(str 42)"), "42");
}

// Phase 2: Comparison tensor output (NumPy-style)

#[test]
fn test_elem_eq_vector() {
    assert_eq!(eval("(== [1 2] [1 3])"), "[ True False]");
}

#[test]
fn test_elem_eq_broadcast() {
    assert_eq!(eval("(== 1 [1 1 2])"), "[ True  True False]");
}

#[test]
fn test_neq_vector() {
    assert_eq!(eval("(!= [1 2] [1 3])"), "[False  True]");
}

#[test]
fn test_lt_vector() {
    assert_eq!(eval("(< [1 5] 3)"), "[ True False]");
}

#[test]
fn test_le_vector() {
    assert_eq!(eval("(<= [1 2 3] 2)"), "[ True  True False]");
}

// Phase 2: case form

#[test]
fn test_case() {
    assert_eq!(eval("(case 2  1 :low  2 :mid  3 :high  :unknown)"), ":mid");
}

// Phase 3: Easy builtins

#[test]
fn test_tensor_from_list() {
    assert_eq!(eval("(tensor '[1 2 3])"), "[1. 2. 3.]");
}

#[test]
fn test_tensor_from_float_list() {
    assert_eq!(eval("(tensor '(1.0 2.0 3.0))"), "[1. 2. 3.]");
}

#[test]
fn test_tensor_passthrough() {
    assert_eq!(eval("(tensor [1 2 3])"), "[1. 2. 3.]");
}

#[test]
fn test_range_1arg() {
    assert_eq!(eval("(range 5)"), "[0 1 2 3 4]");
}

#[test]
fn test_range_3arg() {
    assert_eq!(eval("(range 10 25 5)"), "[10 15 20]");
}

#[test]
fn test_swapaxes_shape() {
    assert_eq!(eval("(shape (swapaxes [[1 2 3] [4 5 6]] 0 1))"), "[3. 2.]");
}

#[test]
fn test_var_1d() {
    let result = eval("(var [1 2 3])");
    let v: f64 = result.parse().unwrap();
    assert!((v - 0.6666667).abs() < 1e-5);
}

#[test]
fn test_normalize_1d() {
    assert_eq!(eval("(normalize [1 2 3])"), "[0.16666667 0.33333334        0.5]");
}

#[test]
fn test_index_of_found() {
    assert_eq!(eval("(index-of '(10 20 30 40) 30)"), "2");
}

#[test]
fn test_index_of_not_found() {
    assert_eq!(eval("(index-of '(10 20 30) 99)"), "-1");
}

#[test]
fn test_find_found() {
    assert_eq!(eval("(find (fn [x] (> x 3)) '(1 2 3 4 5))"), "4");
}

#[test]
fn test_find_not_found() {
    assert_eq!(eval("(find (fn [x] (> x 10)) '(1 2 3))"), "nil");
}

#[test]
fn test_symbol_q_true() {
    assert_eq!(eval("(symbol? 'W)"), "True");
}

#[test]
fn test_symbol_q_false() {
    assert_eq!(eval("(symbol? 42)"), "False");
}

#[test]
fn test_gensym_prefix() {
    let result = eval("(gensym \"var\")");
    assert!(result.starts_with("var"), "gensym should start with prefix: {}", result);
}

// Phase 3: Medium builtins

#[test]
fn test_dynamic_slice() {
    assert_eq!(eval("(dynamic-slice (arange 5) 1 3)"), "[1 2 3]");
}

#[test]
fn test_mse_loss() {
    let result = eval("(mse-loss [1.0 2.0 3.0] [1.1 1.9 3.1])");
    let v: f64 = result.parse().unwrap();
    assert!((v - 0.01).abs() < 1e-5, "mse-loss: got {}", v);
}

#[test]
fn test_mae_loss() {
    let result = eval("(mae-loss [1.0 2.0 3.0] [1.2 1.8 3.1])");
    let v: f64 = result.parse().unwrap();
    assert!((v - 0.16666667).abs() < 1e-5, "mae-loss: got {}", v);
}

#[test]
fn test_sparse_cross_entropy() {
    let result = eval("(sparse-cross-entropy [[0.9 0.1] [0.2 0.8]] [0 1] :i32)");
    let v: f64 = result.parse().unwrap();
    assert!((v - 0.40429434).abs() < 1e-5, "sparse-cross-entropy: got {}", v);
}

#[test]
fn test_tree_map_zeros() {
    assert_eq!(
        eval("(tree-map-zeros {:a 1.0 :b [2.0 3.0]})"),
        "{:a 0.0, :b [0. 0.]}"
    );
}

#[test]
fn test_tree_map_lambda() {
    assert_eq!(
        eval("(tree-map (fn [x] (* x 2.0)) {:a 1.0 :b 2.0})"),
        "{:a 2.0, :b 4.0}"
    );
}

#[test]
fn test_tree_map_nested() {
    assert_eq!(
        eval("(tree-map (fn [x] (* x x)) {:layer1 {:w [2.0 4.0] :b 0.5}})"),
        "{:layer1 {:b 0.25, :w [ 4. 16.]}}"
    );
}

#[test]
fn test_tree_reduce_dict() {
    assert_eq!(eval("(tree-reduce + {:a 1 :b 2 :c 3} 0)"), "6");
}

#[test]
fn test_tree_reduce_list() {
    assert_eq!(eval("(tree-reduce * '(2 3 4) 1)"), "24");
}

#[test]
fn test_flatten_leaves() {
    let result = eval("(first (flatten {:a 1.0 :b 2.0}))");
    assert!(result.contains("1.0") && result.contains("2.0"), "flatten: got {}", result);
}

// Guard tests

#[test]
fn test_guard_no_nan_passes() {
    assert_eq!(eval("(guard :no-nan [1.0 2.0 3.0])"), "[1. 2. 3.]");
}

#[test]
fn test_guard_shape_passes() {
    assert_eq!(eval("(guard :shape [3] [1.0 2.0 3.0])"), "[1. 2. 3.]");
}

#[test]
fn test_guard_range_passes() {
    assert_eq!(eval("(guard :range [0.0 10.0] [1.0 5.0 9.0])"), "[1. 5. 9.]");
}

#[test]
fn test_guard_no_nan_breach() {
    // NaN breach causes process::exit(1), so test via subprocess
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_sheaf"))
        .args(["-c", "(guard :no-nan [1.0 (/ 0.0 0.0)])"])
        .output()
        .expect("failed to run sheaf");
    assert!(!output.status.success(), "Expected exit code 1 for NaN breach");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Guard Breached"), "Expected breach message, got: {}", stderr);
}

#[test]
fn test_guard_shape_breach() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_sheaf"))
        .args(["-c", "(guard :shape [5] [1.0 2.0 3.0])"])
        .output()
        .expect("failed to run sheaf");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Shape mismatch"), "Expected shape mismatch, got: {}", stderr);
}

#[test]
fn test_guard_range_breach() {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_sheaf"))
        .args(["-c", "(guard :range [0.0 5.0] [1.0 10.0])"])
        .output()
        .expect("failed to run sheaf");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Guard Breached"), "Expected breach message, got: {}", stderr);
}

// Tracing test: eval_source_with_tracing should not crash

#[test]
fn test_tracing_does_not_crash() {
    use sheaf_compiler::interpreter::eval::eval_source_with_tracing;
    use sheaf_compiler::interpreter::tracer::{LogFormat, TraceLevel, TracerConfig};

    let config = TracerConfig {
        enabled: true,
        scope_filter: None,
        level: TraceLevel::Normal,
        format: LogFormat::Console,
        cli_guards: Vec::new(),
    };
    let source = "(defn f [x] (+ x 1.0))\n(f [1.0 2.0])";
    let result = eval_source_with_tracing(source, None, config);
    assert!(result.is_ok(), "tracing eval failed: {:?}", result.err());
}
