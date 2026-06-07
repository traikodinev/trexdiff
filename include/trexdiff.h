#ifndef TREXDIFF_H
#define TREXDIFF_H

#include <stdlib.h>
#include <stdbool.h>
#include "tensor2d.h"

// op type
typedef enum {
    OP_NOOP,
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_RELU,
    OP_SIGMOID,
    OP_LN,
    OP_TRANSPOSE,
    OP_BROADCAST_ADD,
    OP_BROADCAST_SUB,
    OP_SIN,
    OP_COS
} OpType;


// Reverse mode autodiff node
typedef struct Node Node;

// Node array - for graph representation
typedef struct NodeArray {
    Node** graph;
    size_t size;
} NodeArray;


struct Node {
    Tensor2D* val;
    Tensor2D* grad; // dL/d_Node
    Tensor2D* partial; // partial accumulated in the current backward pass
                    // this lets us call .backward multiple times to accumulate losses

    bool visited; // node visted (in a pass regardless of direction)
    NodeArray* topo_graph;

    Node** inputs;
    unsigned short input_count;

    OpType op_type;
};


Node* init(Tensor2D* val);
Node* combine(Node* a, Node* b, OpType op_type);

// TODO: better representation of functionals
Node *transform(Node *a, OpType op_type);

static inline Node* add(Node* a, Node* b) {
    return combine(a, b, OP_ADD);
}

static inline Node* sub(Node* a, Node* b) {
    return combine(a, b, OP_SUB);
}

static inline Node* mul(Node* a, Node* b) {
    return combine(a, b, OP_MUL);
}

static inline Node* broadcast_add(Node* a, Node* b) {
    return combine(a, b, OP_BROADCAST_ADD);
}

static inline Node* broadcast_sub(Node* a, Node* b) {
    return combine(a, b, OP_BROADCAST_SUB);
}

static inline Node* relu(Node* a) {
    return transform(a, OP_RELU);
}

static inline Node* sigmoid(Node* a) {
    return transform(a, OP_SIGMOID);
}

static inline Node* trex_log(Node* a) {
    return transform(a, OP_LN);
}

Node* transpose(Node* a);

void free_node(Node *n);
void free_node_shallow(Node *n);
void reset_visited(Node *z);
int reset_and_count(Node *z);

void zerograd(Node *z);
void forward(Node *z);
int backward(Node *z, Tensor2D* partial);


Tensor2D* finite_diff(Node *input, Node *target);

#endif
