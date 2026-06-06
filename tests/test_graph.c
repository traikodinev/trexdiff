#include <stdio.h>
#include <math.h>
#include "minunit.h"
#include "trexdiff.h"

int tests_run = 0;

#define EPSILON 1e-5

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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


// Simple dot product: W(1x2) @ x(2x1) -> 1x1
static char *test_matmul_forward(void) {
    double w_data[] = {0.5, 1.0};
    double x_data[] = {2.0, 3.0};

    Node *W    = init(tensor2d_from_array(1, 2, w_data));
    Node *x    = init(tensor2d_from_array(2, 1, x_data));
    Node *loss = mul(W, x);

    forward(loss);

    // 0.5*2.0 + 1.0*3.0 = 4.0
    mu_assert_is_scalar("matmul_forward: loss is 1x1", loss->val);
    mu_assert_tensor_close_scalar("matmul_forward: value", loss->val, 4.0);

    free_node(loss);
    return 0;
}

static char *test_matmul_backward(void) {
    double w_data[] = {0.5, 1.0};
    double x_data[] = {2.0, 3.0};

    Node *W    = init(tensor2d_from_array(1, 2, w_data));
    Node *x    = init(tensor2d_from_array(2, 1, x_data));
    Node *loss = mul(W, x);

    forward(loss);
    backward(loss, tensor2d_ones(1, 1));

    mu_assert_is_scalar("matmul_backward: loss is 1x1", loss->val);
    mu_assert_tensor_close("matmul_backward: dW", W->grad, finite_diff(W, loss));
    mu_assert_tensor_close("matmul_backward: dx", x->grad, finite_diff(x, loss));

    free_node(loss);
    return 0;
}

// Linear layer with sigmoid activation:
//   loss = a(1x2) @ sigmoid(W(2x2) @ x(2x1) + b(2x1))  ->  1x1
static char *test_matrix_linear(void) {
    double W_data[] = { 0.5,  1.0,
                       -0.5,  2.0};
    double x_data[] = {1.5, -1.0};
    double b_data[] = {0.1,  0.2};
    double a_data[] = {1.0,  1.0};

    Node *W = init(tensor2d_from_array(2, 2, W_data));
    Node *x = init(tensor2d_from_array(2, 1, x_data));
    Node *b = init(tensor2d_from_array(2, 1, b_data));
    Node *a = init(tensor2d_from_array(1, 2, a_data));

    Node *Wx   = mul(W, x);      // 2x2 @ 2x1 = 2x1
    Node *Wx_b = add(Wx, b);     // 2x1 + 2x1 = 2x1
    Node *h    = sigmoid(Wx_b);  // elementwise, 2x1
    Node *loss = mul(a, h);      // 1x2 @ 2x1 = 1x1

    forward(loss);
    backward(loss, tensor2d_ones(1, 1));

    mu_assert_is_scalar("matrix_linear: loss is 1x1", loss->val);
    mu_assert_tensor_close("matrix_linear: dW", W->grad, finite_diff(W, loss));
    mu_assert_tensor_close("matrix_linear: dx", x->grad, finite_diff(x, loss));
    mu_assert_tensor_close("matrix_linear: db", b->grad, finite_diff(b, loss));
    mu_assert_tensor_close("matrix_linear: da", a->grad, finite_diff(a, loss));

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
    mu_run_test(test_matmul_forward);
    mu_run_test(test_matmul_backward);
    mu_run_test(test_matrix_linear);
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

