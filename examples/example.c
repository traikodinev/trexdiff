#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "trexdiff.h"


void test(void) {
    // More complex graph:
    //
    //  a --
    //       mul1 = a * b --
    //  b --               add2 = mul1 + mul2 --
    //                                            mul3 = add2 * add1 => loss
    //  c --               add1 = c + d ---------
    //       mul2 = c * d --
    //  d --
    //
    Node* a = init(1.5);
    Node* b = init(2.0);
    Node* c = init(0.5);
    Node* d = init(3.0);

    Node* mul1 = mul(a, b);           // a * b
    Node* mul2 = mul(c, d);           // c * d
    Node* add1 = add(c, d);           // c + d
    Node* add2 = add(mul1, mul2);     // (a*b) + (c*d)
    Node* loss = mul(add2, add1);     // ((a*b)+(c*d)) * (c+d)

    forward(loss);
    backward(loss, 1.0);

    printf("\n--- test ---\n");
    printf("loss: %f\n", loss->val);
    printf("grad:  da=%f, db=%f, dc=%f, dd=%f\n",
           a->grad, b->grad, c->grad, d->grad);

    double da_fd = finite_diff(a, loss);
    double db_fd = finite_diff(b, loss);
    double dc_fd = finite_diff(c, loss);
    double dd_fd = finite_diff(d, loss);
    printf("fd:    da=%f, db=%f, dc=%f, dd=%f\n",
           da_fd, db_fd, dc_fd, dd_fd);
}


int main(void) {
    // x
    //    -- c = x + y
    // y          |         ---- e = c x d  => loss
    //          d = c + z
    // z   -- 
    // 
    // Node* x = init(2);
    // Node* y = init(0.5);
    // Node* z = init(3);
    
    // Node* c = add(x, y);
    // Node* d = add(c, z);
    
    // Node* e = mul(c, d);
    

    // forward(e);
    // backward(e, 1.0);
    
    // printf("e: %f.\n", e->val);
    // printf("dx, dy, dz: %f., %f., %f.\n", x->grad, y->grad, z->grad);


    // double dx_fd = finite_diff(x, e);
    // double dy_fd = finite_diff(y, e);
    // double dz_fd = finite_diff(z, e);
    // printf("dx, dy, dz: %f., %f., %f.\n", dx_fd, dy_fd, dz_fd);

    test();
}
