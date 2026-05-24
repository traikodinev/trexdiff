#ifndef TREXDIFF_H
#define TREXDIFF_H

#include <stdlib.h>
#include <stdbool.h>

// op type
typedef enum {
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_RELU,
    OP_NOOP
} OpType;


// Reverse mode autodiff node
typedef struct Node Node;
struct Node {
    double val;
    double grad; // dL/d_Node

    bool visited; // node visted (in a pass regardless of direction)

    Node** inputs;
    short input_count;

    OpType op_type;
};


Node* init(double val);
Node* combine(Node* a, Node* b, OpType op_type);


static inline Node* add(Node* a, Node* b) {
    return combine(a, b, OP_ADD);
}

static inline Node* mul(Node* a, Node* b) {
    return combine(a, b, OP_MUL);
}


void free_node(Node *n);
void reset_visited(Node *z);
void zerograd(Node *z);
void forward(Node *z);
void backward(Node *z, double partial);


double finite_diff(Node *input, Node *target);

#endif
