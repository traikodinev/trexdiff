#include <stdio.h>
#include <math.h>
#include "minunit.h"
#include "trexdiff.h"

int tests_run = 0;

#define EPSILON 1e-4

// helpers

// Create a 1x1 tensor from a scalar value.
static Tensor2D* sc(double v) {
    double arr[1] = {v};
    return tensor2d_from_array(1, 1, arr);
}

// Element-wise comparison of two tensors with tolerance.
static bool tensors_close(Tensor2D* a, Tensor2D* b) {
    if (a->M != b->M || a->N != b->N) return false;
    for (size_t i = 0; i < a->M * a->N; i++) {
        if (fabs(a->matrix[i] - b->matrix[i]) >= EPSILON) return false;
    }
    return true;
}

// Compare a 1x1 tensor against a scalar value.
static bool tensor_close_scalar(Tensor2D* a, double v) {
    return a->M == 1 && a->N == 1 && fabs(a->matrix[0] - v) < EPSILON;
}

#define mu_assert_tensor_close(msg, a, b) \
    mu_assert(msg, tensors_close(a, b))

#define mu_assert_tensor_close_scalar(msg, a, v) \
    mu_assert(msg, tensor_close_scalar(a, v))

#define mu_assert_is_scalar(msg, t) \
    mu_assert(msg, (t)->M == 1 && (t)->N == 1)


// Graph:
//
//  a --
//       mul1 = a * b --
//  b --               add2 = mul1 + mul2 --
//                                            loss = add2 * add1
//  c --               add1 = c + d ---------
//       mul2 = c * d --
//  d --

static char *test_forward(void) {
    Node *a = init(sc(1.5));
    Node *b = init(sc(2.0));
    Node *c = init(sc(0.5));
    Node *d = init(sc(3.0));

    Node *mul1 = mul(a, b);
    Node *mul2 = mul(c, d);
    Node *add1 = add(c, d);
    Node *add2 = add(mul1, mul2);
    Node *loss = mul(add2, add1);

    forward(loss);

    // ((1.5*2.0) + (0.5*3.0)) * (0.5+3.0) = (3.0+1.5) * 3.5 = 15.75
    mu_assert_is_scalar("forward: loss is 1x1", loss->val);
    mu_assert_tensor_close_scalar("forward: loss != 15.75", loss->val, 15.75);

    free_node(loss);
    return 0;
}

static char *test_backward(void) {
    Node *a = init(sc(1.5));
    Node *b = init(sc(2.0));
    Node *c = init(sc(0.5));
    Node *d = init(sc(3.0));

    Node *mul1 = mul(a, b);
    Node *mul2 = mul(c, d);
    Node *add1 = add(c, d);
    Node *add2 = add(mul1, mul2);
    Node *loss = mul(add2, add1);

    forward(loss);
    backward(loss, tensor2d_ones(1, 1));

    mu_assert_is_scalar("backward: loss is 1x1", loss->val);
    mu_assert_tensor_close("backward: da", a->grad, finite_diff(a, loss));
    mu_assert_tensor_close("backward: db", b->grad, finite_diff(b, loss));
    mu_assert_tensor_close("backward: dc", c->grad, finite_diff(c, loss));
    mu_assert_tensor_close("backward: dd", d->grad, finite_diff(d, loss));

    free_node(loss);
    return 0;
}

static char *test_squared_loss(void) {
    Node *x = init(sc(1.5));
    Node *a = init(sc(2.0));
    Node *b = init(sc(0.5));
    Node *y = init(sc(10.0));

    Node *diff = sub(y, add(mul(a, x), b));
    Node *loss = mul(diff, diff);

    forward(loss);
    backward(loss, tensor2d_ones(1, 1));

    mu_assert_is_scalar("squared loss: loss is 1x1", loss->val);
    mu_assert_tensor_close("squared loss: da",    a->grad,    finite_diff(a,    loss));
    mu_assert_tensor_close("squared loss: db",    b->grad,    finite_diff(b,    loss));
    mu_assert_tensor_close("squared loss: dx",    x->grad,    finite_diff(x,    loss));
    mu_assert_tensor_close("squared loss: dy",    y->grad,    finite_diff(y,    loss));
    mu_assert_tensor_close("squared loss: ddiff", diff->grad, finite_diff(diff, loss));

    free_node(loss);
    return 0;
}

static char *test_nonlinear(void) {
    Node *x = init(sc(1.5));
    Node *a = init(sc(2.0));
    Node *b = init(sc(0.5));
    Node *y = init(sc(10.0));

    Node *diff = sub(y, relu(add(mul(a, x), b)));
    Node *loss = mul(diff, diff);

    forward(loss);
    backward(loss, tensor2d_ones(1, 1));

    mu_assert_is_scalar("nonlinear: loss is 1x1", loss->val);
    mu_assert_tensor_close("nonlinear: da",    a->grad,    finite_diff(a,    loss));
    mu_assert_tensor_close("nonlinear: db",    b->grad,    finite_diff(b,    loss));
    mu_assert_tensor_close("nonlinear: dx",    x->grad,    finite_diff(x,    loss));
    mu_assert_tensor_close("nonlinear: dy",    y->grad,    finite_diff(y,    loss));
    mu_assert_tensor_close("nonlinear: ddiff", diff->grad, finite_diff(diff, loss));

    free_node(loss);
    return 0;
}

static char *test_sigmoid(void) {
    Node *x     = init(sc(1.5));
    Node *a     = init(sc(2.0));
    Node *b     = init(sc(0.5));
    Node *scale = init(sc(7.0));

    // to avoid multiply-by-1 accidents
    Node *prob = mul(scale, sigmoid(add(mul(a, x), b)));
    Node *loss = trex_log(prob);

    forward(loss);
    backward(loss, tensor2d_ones(1, 1));

    mu_assert_is_scalar("sigmoid: loss is 1x1", loss->val);
    mu_assert_is_scalar("sigmoid: prob is 1x1", prob->val);
    mu_assert_tensor_close_scalar("sigmoid: prob value", prob->val, 7.0 * 0.97068776924);
    mu_assert_tensor_close("sigmoid: da", a->grad, finite_diff(a, loss));
    mu_assert_tensor_close("sigmoid: db", b->grad, finite_diff(b, loss));
    mu_assert_tensor_close("sigmoid: dx", x->grad, finite_diff(x, loss));

    free_node(loss);
    return 0;
}

static char *test_accumulation(void) {
    Node *x = init(sc(1.5));
    Node *a = init(sc(2.0));
    Node *b = init(sc(0.5));
    Node *y = init(sc(10.0));

    Node *diff = sub(y, add(mul(a, x), b));
    Node *loss = mul(diff, diff);

    forward(loss);
    backward(loss, tensor2d_ones(1, 1));
    backward(loss, tensor2d_ones(1, 1));

    mu_assert_tensor_close("accumulation: da",    a->grad,    tensor2d_scalar_mul(finite_diff(a,    loss), 2.0));
    mu_assert_tensor_close("accumulation: db",    b->grad,    tensor2d_scalar_mul(finite_diff(b,    loss), 2.0));
    mu_assert_tensor_close("accumulation: dx",    x->grad,    tensor2d_scalar_mul(finite_diff(x,    loss), 2.0));
    mu_assert_tensor_close("accumulation: dy",    y->grad,    tensor2d_scalar_mul(finite_diff(y,    loss), 2.0));
    mu_assert_tensor_close("accumulation: ddiff", diff->grad, tensor2d_scalar_mul(finite_diff(diff, loss), 2.0));

    free_node(loss);
    return 0;
}


// matrix tests

// Two matmuls chained: loss = W2(1x2) @ W1(2x3) @ x(3x1)  ->  1x1
static char *test_matmul_chain(void) {
    double W1_data[] = {0.1, 0.2, 0.3,
                        0.4, 0.5, 0.6};
    double x_data[]  = {1.0, -1.0, 0.5};
    double W2_data[] = {0.7, 0.8};

    Node *W1 = init(tensor2d_from_array(2, 3, W1_data));
    Node *x  = init(tensor2d_from_array(3, 1, x_data));
    Node *W2 = init(tensor2d_from_array(1, 2, W2_data));

    Node *h    = mul(W1, x);    // 2x3 @ 3x1 = 2x1
    Node *loss = mul(W2, h);    // 1x2 @ 2x1 = 1x1

    forward(loss);
    backward(loss, tensor2d_ones(1, 1));

    mu_assert_is_scalar("matmul_chain: loss is 1x1", loss->val);
    // h = [0.1-0.2+0.15, 0.4-0.5+0.3] = [0.05, 0.2]
    // loss = 0.7*0.05 + 0.8*0.2 = 0.195
    mu_assert_tensor_close_scalar("matmul_chain: forward", loss->val, 0.195);
    mu_assert_tensor_close("matmul_chain: dW1", W1->grad, finite_diff(W1, loss));
    mu_assert_tensor_close("matmul_chain: dx",  x->grad,  finite_diff(x,  loss));
    mu_assert_tensor_close("matmul_chain: dW2", W2->grad, finite_diff(W2, loss));

    free_node(loss);
    return 0;
}

// Add then subtract column vectors, reduced with a row vector:
//   loss = w(1x3) @ ((a(3x1) + b(3x1)) - c(3x1))  ->  1x1
static char *test_add_sub_vectors(void) {
    double a_data[] = { 1.0, -0.5,  2.0};
    double b_data[] = { 0.3,  0.8, -1.0};
    double c_data[] = { 0.5,  0.2,  0.3};
    double w_data[] = { 1.0,  1.0,  1.0};

    Node *a = init(tensor2d_from_array(3, 1, a_data));
    Node *b = init(tensor2d_from_array(3, 1, b_data));
    Node *c = init(tensor2d_from_array(3, 1, c_data));
    Node *w = init(tensor2d_from_array(1, 3, w_data));

    Node *t1   = add(a, b);     // 3x1
    Node *t2   = sub(t1, c);    // 3x1
    Node *loss = mul(w, t2);    // 1x3 @ 3x1 = 1x1

    forward(loss);
    backward(loss, tensor2d_ones(1, 1));

    mu_assert_is_scalar("add_sub_vectors: loss is 1x1", loss->val);
    mu_assert_tensor_close("add_sub_vectors: da", a->grad, finite_diff(a, loss));
    mu_assert_tensor_close("add_sub_vectors: db", b->grad, finite_diff(b, loss));
    mu_assert_tensor_close("add_sub_vectors: dc", c->grad, finite_diff(c, loss));
    mu_assert_tensor_close("add_sub_vectors: dw", w->grad, finite_diff(w, loss));

    free_node(loss);
    return 0;
}

// ReLU on a vector with mixed-sign values:
//   loss = w(1x4) @ relu(x(4x1))  ->  1x1
static char *test_relu_vector(void) {
    double x_data[] = { 1.5, -2.0,  0.3, -0.1};
    double w_data[] = { 0.5,  1.0,  1.5,  2.0};

    Node *x = init(tensor2d_from_array(4, 1, x_data));
    Node *w = init(tensor2d_from_array(1, 4, w_data));

    Node *h    = relu(x);       // 4x1
    Node *loss = mul(w, h);     // 1x4 @ 4x1 = 1x1

    forward(loss);
    backward(loss, tensor2d_ones(1, 1));

    mu_assert_is_scalar("relu_vector: loss is 1x1", loss->val);
    // relu([1.5,-2,0.3,-0.1]) = [1.5,0,0.3,0]
    // loss = 0.5*1.5 + 0 + 1.5*0.3 + 0 = 0.75 + 0.45 = 1.2
    mu_assert_tensor_close_scalar("relu_vector: forward", loss->val, 1.2);
    mu_assert_tensor_close("relu_vector: dx", x->grad, finite_diff(x, loss));
    mu_assert_tensor_close("relu_vector: dw", w->grad, finite_diff(w, loss));

    free_node(loss);
    return 0;
}

// Sigmoid on a vector:
//   loss = w(1x3) @ sigmoid(x(3x1))  ->  1x1
static char *test_sigmoid_vector(void) {
    double x_data[] = { 0.5, -1.0,  1.5};
    double w_data[] = { 1.0,  2.0,  0.5};

    Node *x = init(tensor2d_from_array(3, 1, x_data));
    Node *w = init(tensor2d_from_array(1, 3, w_data));

    Node *h    = sigmoid(x);    // 3x1
    Node *loss = mul(w, h);     // 1x3 @ 3x1 = 1x1

    forward(loss);
    backward(loss, tensor2d_ones(1, 1));

    mu_assert_is_scalar("sigmoid_vector: loss is 1x1", loss->val);
    mu_assert_tensor_close("sigmoid_vector: dx", x->grad, finite_diff(x, loss));
    mu_assert_tensor_close("sigmoid_vector: dw", w->grad, finite_diff(w, loss));

    free_node(loss);
    return 0;
}

// Log on a positive vector:
//   loss = w(1x3) @ ln(x(3x1))  ->  1x1
static char *test_ln_vector(void) {
    double x_data[] = {1.0, 2.0, 0.5};
    double w_data[] = {1.0, 1.0, 1.0};

    Node *x = init(tensor2d_from_array(3, 1, x_data));
    Node *w = init(tensor2d_from_array(1, 3, w_data));

    Node *h    = trex_log(x);   // 3x1
    Node *loss = mul(w, h);     // 1x3 @ 3x1 = 1x1

    forward(loss);
    backward(loss, tensor2d_ones(1, 1));

    mu_assert_is_scalar("ln_vector: loss is 1x1", loss->val);
    // ln(1) + ln(2) + ln(0.5) = 0 + ln(2) - ln(2) = 0
    mu_assert_tensor_close_scalar("ln_vector: forward", loss->val, 0.0);
    mu_assert_tensor_close("ln_vector: dx", x->grad, finite_diff(x, loss));
    mu_assert_tensor_close("ln_vector: dw", w->grad, finite_diff(w, loss));

    free_node(loss);
    return 0;
}

// Two-layer MLP with ReLU hidden layer:
//   loss = W2(1x4) @ relu(W1(4x3) @ x(3x1) + b(4x1))  ->  1x1
static char *test_two_layer(void) {
    double W1_data[] = { 0.5, -0.3,  0.8,
                         0.1,  0.9, -0.4,
                        -0.6,  0.2,  0.7,
                         0.3, -0.5,  0.1};
    double b_data[]  = { 0.1, -0.2,  0.3, -0.1};
    double x_data[]  = { 1.0, -0.5,  0.8};
    double W2_data[] = { 0.4,  0.6, -0.3,  0.7};

    Node *W1 = init(tensor2d_from_array(4, 3, W1_data));
    Node *b  = init(tensor2d_from_array(4, 1, b_data));
    Node *x  = init(tensor2d_from_array(3, 1, x_data));
    Node *W2 = init(tensor2d_from_array(1, 4, W2_data));

    Node *z    = add(mul(W1, x), b);   // 4x3@3x1 + 4x1 = 4x1
    Node *h    = relu(z);              // 4x1
    Node *loss = mul(W2, h);           // 1x4 @ 4x1 = 1x1

    forward(loss);
    backward(loss, tensor2d_ones(1, 1));

    mu_assert_is_scalar("two_layer: loss is 1x1", loss->val);
    mu_assert_tensor_close("two_layer: dW1", W1->grad, finite_diff(W1, loss));
    mu_assert_tensor_close("two_layer: db",  b->grad,  finite_diff(b,  loss));
    mu_assert_tensor_close("two_layer: dx",  x->grad,  finite_diff(x,  loss));
    mu_assert_tensor_close("two_layer: dW2", W2->grad, finite_diff(W2, loss));

    free_node(loss);
    return 0;
}

// Gradient accumulation with matrix inputs: two backward calls -> 2x gradients
static char *test_accumulation_matrix(void) {
    double W_data[] = { 0.5,  1.0, -0.5};
    double x_data[] = { 2.0, -1.0,  0.5};

    Node *W    = init(tensor2d_from_array(1, 3, W_data));
    Node *x    = init(tensor2d_from_array(3, 1, x_data));
    Node *loss = mul(W, x);    // 1x3 @ 3x1 = 1x1

    forward(loss);
    backward(loss, tensor2d_ones(1, 1));
    backward(loss, tensor2d_ones(1, 1));

    mu_assert_tensor_close("accumulation_matrix: dW", W->grad,
                           tensor2d_scalar_mul(finite_diff(W, loss), 2.0));
    mu_assert_tensor_close("accumulation_matrix: dx", x->grad,
                           tensor2d_scalar_mul(finite_diff(x, loss), 2.0));

    free_node(loss);
    return 0;
}


static char *test_transpose(void) {
    double x_data[] = {1.0, 2.0, 3.0};
    Node *x = init(tensor2d_from_array(1, 3, x_data));
    Node *loss = mul(x, transpose(x)); // 1x3 @ 3x1 = 1x1

    forward(loss);
    backward(loss, tensor2d_ones(1, 1));

    mu_assert_tensor_close("transpose: dx", x->grad, finite_diff(x, loss));

    free_node(loss);
    return 0;
}


static char *test_broadcast(void) {
    double x_data[] = {1.0, 2.0, 3.0};
    double y_data[] = {0.5, 1.5, -1.0};

    Node *x = init(tensor2d_from_array(1, 3, x_data));
    Node *y = init(tensor2d_from_array(1, 3, y_data));
    Node *loss = mul(
        broadcast_add(x, init(sc(1.0))),
        transpose(broadcast_sub(y, init(sc(2.0))))
    ); // (x+1) @ (y-2) -> 1x1


    forward(loss);
    backward(loss, tensor2d_ones(1, 1));

    mu_assert_tensor_close("broadcast: dx", x->grad, finite_diff(x, loss));
    mu_assert_tensor_close("broadcast: dy", y->grad, finite_diff(y, loss));

    free_node(loss);
    return 0;
}


static char *test_trig(void) {
    double x_data[] = {0.0, 1.0, -4.3};
    Node *x = init(tensor2d_from_array(1, 3, x_data));
    Node *loss = mul(transform(x, OP_SIN), transpose(transform(x, OP_COS)));

    forward(loss);
    backward(loss, tensor2d_ones(1, 1));

    mu_assert_tensor_close("trig: dx", x->grad, finite_diff(x, loss));

    free_node(loss);
    return 0;
}

static char *test_categorical_cross_entropy(void) {
    double logits_data[] = {1.0, 2.0, 0.5};
    double target_data[] = {0.0, 1.0, 0.0};

    // Shape: 1 sample x 3 classes
    Node *logits = init(tensor2d_from_array(1, 3, logits_data));
    Node *target = init(tensor2d_from_array(1, 3, target_data));
    Node *loss = categorical_cross_entropy(logits, target);

    forward(loss);
    backward(loss, tensor2d_ones(1, 1));

    double max_logit = 2.0;
    double log_sum_exp = max_logit
        + log(exp(1.0 - max_logit) + exp(2.0 - max_logit) + exp(0.5 - max_logit));
    double expected = log_sum_exp - 2.0;

    mu_assert_is_scalar("categorical_cross_entropy: loss is 1x1", loss->val);
    mu_assert_tensor_close_scalar("categorical_cross_entropy: forward", loss->val, expected);
    mu_assert_tensor_close("categorical_cross_entropy: dlogits", logits->grad, finite_diff(logits, loss));

    free_node(loss);
    return 0;
}

static char *test_categorical_cross_entropy_batched(void) {
    // Shape: 3 samples x 2 classes
    double logits_data[] = {
        1.0, 2.0,
        0.5, -1.0,
        3.0, 0.0
    };

    double target_data[] = {
        0.0, 1.0,  // class 1
        1.0, 0.0,  // class 0
        1.0, 0.0   // class 0
    };

    Node *logits = init(tensor2d_from_array(3, 2, logits_data));
    Node *target = init(tensor2d_from_array(3, 2, target_data));
    Node *loss = categorical_cross_entropy(logits, target);

    forward(loss);
    backward(loss, tensor2d_ones(1, 1));

    double expected = 0.0;

    // sample 0: logits [1.0, 2.0], target class 1
    {
        double max_logit = 2.0;
        double log_sum_exp = max_logit
            + log(exp(1.0 - max_logit) + exp(2.0 - max_logit));
        expected += log_sum_exp - 2.0;
    }

    // sample 1: logits [0.5, -1.0], target class 0
    {
        double max_logit = 0.5;
        double log_sum_exp = max_logit
            + log(exp(0.5 - max_logit) + exp(-1.0 - max_logit));
        expected += log_sum_exp - 0.5;
    }

    // sample 2: logits [3.0, 0.0], target class 0
    {
        double max_logit = 3.0;
        double log_sum_exp = max_logit
            + log(exp(3.0 - max_logit) + exp(0.0 - max_logit));
        expected += log_sum_exp - 3.0;
    }

    expected /= 3.0;

    mu_assert_is_scalar("categorical_cross_entropy_batched: loss is 1x1", loss->val);
    mu_assert_tensor_close_scalar("categorical_cross_entropy_batched: forward", loss->val, expected);
    mu_assert_tensor_close("categorical_cross_entropy_batched: dlogits", logits->grad, finite_diff(logits, loss));

    free_node(loss);
    return 0;
}


// Runner
static char *all_tests(void) {
    mu_run_test(test_forward);
    mu_run_test(test_backward);
    mu_run_test(test_squared_loss);
    mu_run_test(test_accumulation);
    mu_run_test(test_nonlinear);
    mu_run_test(test_sigmoid);
    mu_run_test(test_matmul_chain);
    mu_run_test(test_add_sub_vectors);
    mu_run_test(test_relu_vector);
    mu_run_test(test_sigmoid_vector);
    mu_run_test(test_ln_vector); 
    mu_run_test(test_two_layer);
    mu_run_test(test_accumulation_matrix);
    mu_run_test(test_transpose);
    mu_run_test(test_broadcast);
    mu_run_test(test_trig);
    mu_run_test(test_categorical_cross_entropy);
    mu_run_test(test_categorical_cross_entropy_batched);
    return 0;
}

int main(void) {
    char *result = all_tests();
    if (result != 0) {
        printf("FAILED: %s\n", result);
    } else {
        printf("ALL TESTS PASSED\n");
    }
    printf("Tests run: %d\n", tests_run);
    return result != 0;
}
