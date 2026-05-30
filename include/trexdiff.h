#ifndef TREXDIFF_H
#define TREXDIFF_H

#include <stdlib.h>
#include <stdbool.h>

// op type
typedef enum {
    OP_NOOP,
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_RELU,
    OP_SIGMOID,
    OP_LN
} OpType;


// Reverse mode autodiff node
typedef struct Node Node;

// Node array - for graph representation
typedef struct NodeArray {
    Node** graph;
    size_t size;
} NodeArray;


struct Node {
    double val;
    double grad; // dL/d_Node
    double partial; // partial accumulated in the current backward pass
                    // this lets us call .backward multiple times to accumulate losses

    bool visited; // node visted (in a pass regardless of direction)
    NodeArray* topo_graph;

    Node** inputs;
    unsigned short input_count;

    OpType op_type;
};


Node* init(double val);
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

static inline Node* relu(Node* a) {
    return transform(a, OP_RELU);
}

static inline Node* sigmoid(Node* a) {
    return transform(a, OP_SIGMOID);
}

static inline Node* trex_log(Node* a) {
    return transform(a, OP_LN);
}


void free_node(Node *n);
void reset_visited(Node *z);
int reset_and_count(Node *z);

void zerograd(Node *z);
void forward(Node *z);
void backward(Node *z, double partial);


double finite_diff(Node *input, Node *target);

#endif
