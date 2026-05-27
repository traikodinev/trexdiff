#include <stdio.h>
#include <math.h>
#include "minunit.h"
#include "trexdiff.h"

int tests_run = 0;

#define EPSILON 1e-4

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
    Node *a = init(1.5);
    Node *b = init(2.0);
    Node *c = init(0.5);
    Node *d = init(3.0);

    Node *mul1 = mul(a, b);
    Node *mul2 = mul(c, d);
    Node *add1 = add(c, d);
    Node *add2 = add(mul1, mul2);
    Node *loss = mul(add2, add1);

    forward(loss);

    // ((1.5*2.0) + (0.5*3.0)) * (0.5+3.0) = (3.0+1.5) * 3.5 = 15.75
    mu_assert_close("forward: loss != 15.75", loss->val, 15.75);

    free_node(loss);
    return 0;
}

static char *test_backward(void) {
    Node *a = init(1.5);
    Node *b = init(2.0);
    Node *c = init(0.5);
    Node *d = init(3.0);

    Node *mul1 = mul(a, b);
    Node *mul2 = mul(c, d);
    Node *add1 = add(c, d);
    Node *add2 = add(mul1, mul2);
    Node *loss = mul(add2, add1);

    forward(loss);
    backward(loss, 1.0);

    mu_assert_close("backward: da", a->grad, finite_diff(a, loss));
    mu_assert_close("backward: db", b->grad, finite_diff(b, loss));
    mu_assert_close("backward: dc", c->grad, finite_diff(c, loss));
    mu_assert_close("backward: dd", d->grad, finite_diff(d, loss));

    free_node(loss);
    return 0;
}

static char *test_squared_loss() {
    Node *x = init(1.5);
    Node *a = init(2.0);
    Node *b = init(0.5);
    Node *y = init(10.0);

    Node *diff = sub(y, add(mul(a, x), b));
    Node *loss = mul(diff, diff);

    forward(loss);
    backward(loss, 1.0);

    mu_assert_close("squared loss: da", a->grad, finite_diff(a, loss));
    mu_assert_close("squared loss: db", b->grad, finite_diff(b, loss));
    mu_assert_close("squared loss: dx", x->grad, finite_diff(x, loss));
    mu_assert_close("squared loss: dy", y->grad, finite_diff(y, loss));
    mu_assert_close("squared loss: ddiff", diff->grad, finite_diff(diff, loss));


    free_node(loss);
    return 0;    
}


static char *test_accumulation() {
    Node *x = init(1.5);
    Node *a = init(2.0);
    Node *b = init(0.5);
    Node *y = init(10.0);

    Node *diff = sub(y, add(mul(a, x), b));
    Node *loss = mul(diff, diff);

    forward(loss);
    backward(loss, 1.0);
    backward(loss, 1.0);

    mu_assert_close("accumulation: da", a->grad, 2 * finite_diff(a, loss));
    mu_assert_close("accumulation: db", b->grad, 2 * finite_diff(b, loss));
    mu_assert_close("accumulation: dx", x->grad, 2 * finite_diff(x, loss));
    mu_assert_close("accumulation: dy", y->grad, 2 * finite_diff(y, loss));
    mu_assert_close("accumulation: ddiff", diff->grad, 2 * finite_diff(diff, loss));

    free_node(loss);
    return 0;    
}


static char *all_tests(void) {
    mu_run_test(test_forward);
    mu_run_test(test_backward);
    mu_run_test(test_squared_loss);
    mu_run_test(test_accumulation);
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
