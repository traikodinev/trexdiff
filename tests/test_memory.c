// Compiled with -fsanitize=leak (see `make test-leak`).
//
// Exercises backward() then frees every node in the graph.
// free_node() does not free the topo_graph allocated during backward(), so
// LeakSanitizer will report the leak and exit non-zero, failing the test.
//
// Expected failures (until the leak is fixed):
//   - loss->topo_graph          (NodeArray*)
//   - loss->topo_graph->graph   (Node**)

#include <stdio.h>
#include "trexdiff.h"

int main(void) {
    // Graph identical to test_graph.c so results are comparable.
    //
    //  a --
    //       mul1 = a * b --
    //  b --               add2 = mul1 + mul2 --
    //                                            loss = add2 * add1
    //  c --               add1 = c + d ---------
    //       mul2 = c * d --
    //  d --

    Node *a    = init(1.5);
    Node *b    = init(2.0);
    Node *c    = init(0.5);
    Node *d    = init(3.0);
    Node *mul1 = mul(a, b);
    Node *mul2 = mul(c, d);
    Node *add1 = add(c, d);
    Node *add2 = add(mul1, mul2);
    Node *loss = mul(add2, add1);

    forward(loss);
    backward(loss, 1.0);   // allocates loss->topo_graph and ->graph

    // Free every node.  loss owns topo_graph but free_node() skips it.
    free_node(a);
    free_node(b);
    free_node(c);
    free_node(d);
    free_node(mul1);
    free_node(mul2);
    free_node(add1);
    free_node(add2);
    free_node(loss);

    // LeakSanitizer runs its check here (at exit) and will report
    // loss->topo_graph as leaked, causing a non-zero exit code.
    return 0;
}
