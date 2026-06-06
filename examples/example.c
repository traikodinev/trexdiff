#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "trexdiff.h"


static Tensor2D* sc(double v) {
    double arr[1] = {v};
    return tensor2d_from_array(1, 1, arr);
}


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
    Node* a = init(sc(1.5));
    Node* b = init(sc(2.0));
    Node* c = init(sc(0.5));
    Node* d = init(sc(3.0));

    Node* mul1 = mul(a, b);           // a * b
    Node* mul2 = mul(c, d);           // c * d
    Node* add1 = add(c, d);           // c + d
    Node* add2 = add(mul1, mul2);     // (a*b) + (c*d)
    Node* loss = mul(add2, add1);     // ((a*b)+(c*d)) * (c+d)

    forward(loss);
    backward(loss, sc(1.0));
}


int main(void) {
    test();
}
