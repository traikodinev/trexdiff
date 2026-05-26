#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "trexdiff.h"


Node* init(double val) {
    Node* node = malloc(sizeof(Node));
    node->input_count = 0;
    node->inputs = NULL;
    node->topo_graph = NULL;
    node->val = val;
    node->grad = 0.0;
    node->op_type = OP_NOOP;
    node->visited = false;

    return node;
}


void free_node(Node *n) {
    free(n->inputs);
    free(n);
}

Node* combine(Node* a, Node* b, OpType op_type) {
    Node* node = init(0.0);
    
    // adding 2 nodes means 2 inputs
    node->input_count = 2;
    node->inputs = malloc(sizeof(Node*) * 2);

    node->inputs[0] = a;
    node->inputs[1] = b;

    node->op_type = op_type;
    node->visited = false;

    return node;
}


void reset_visited(Node *z) {
    z->visited = false;

    if (z -> input_count == 0)
        return;

    reset_visited(z->inputs[0]);
    reset_visited(z->inputs[1]);
}


int reset_and_count(Node *z) {
    // TODO: This is not an exact node count due to multiple visits 
    z->visited = false;

    if (z->input_count == 0)
        return 1;

    return 1 + reset_and_count(z->inputs[0]) + reset_and_count(z->inputs[1]);
}


void zerograd(Node *z) {
    z->grad = 0.0;

    if (z -> input_count == 0)
        return;

    zerograd(z->inputs[0]);
    zerograd(z->inputs[1]);
}


static void _forward(Node* z) {
    if (z -> input_count == 0 || z->visited)
        return;

    _forward(z->inputs[0]);
    _forward(z->inputs[1]);
    
    // TODO: use fwd_complete/bwd_complete to optimize passes
    // if (z->fwd_complete)
    //     return;

    // TODO: multiple input support (maybe?)
    // TODO: function pointers w/ inline (can I?) instead of case 
    switch (z->op_type) {
        case OP_ADD:
            z->val = z->inputs[0]->val + z->inputs[1]->val;
            break;
        case OP_MUL:
            z->val = z->inputs[0]->val * z->inputs[1]->val;
            break;
        case OP_SUB:
            z->val = z->inputs[0]->val - z->inputs[1]->val;
            break;
        case OP_NOOP:
        default:
            // no-op by default
            break;
    }

    z->visited = true;
}


void forward(Node *z) {
    reset_visited(z);
    _forward(z);
}


static inline void _backward(Node *start, double partial) {
    Node* z;
    start->grad += partial;

    for (size_t i = start->topo_graph->size - 1; i > 0; --i) {
        z = start->topo_graph->graph[i];
        
        switch (z->op_type) {
            case OP_ADD:
                z->inputs[0]->grad += z->grad;
                z->inputs[1]->grad += z->grad;
                break;
            case OP_MUL:
                z->inputs[1]->grad += z->grad * z->inputs[0]->val;
                z->inputs[0]->grad += z->grad * z->inputs[1]->val;
                break;
            case OP_SUB:
                z->inputs[0]->grad += z->grad;
                z->inputs[1]->grad += -z->grad;
                break;
            case OP_NOOP:
            default:
                // no-op
                break;
        }
    }
}


static void _topo_sort(Node *z, NodeArray* topo_graph) {
    // TODO: cycle detection
    if (z->visited)
        return;

    if (z->inputs) {
        _topo_sort(z->inputs[0], topo_graph);
        _topo_sort(z->inputs[1], topo_graph);
    }

    z->visited = true;
    topo_graph->graph[topo_graph->size++] = z;
}


void backward(Node *z, double partial) {
    if (z->topo_graph) {
        _backward(z, partial);
        return;
    }

    // TODO: We can optimize this by allowing calls to e.g. `compile`, a la Pytorch
    //  instead of manually computing the topological sorting
    size_t n_nodes = reset_and_count(z);
    z->topo_graph = malloc(sizeof(NodeArray));
    z->topo_graph->graph = malloc(sizeof(Node*) * n_nodes);
    z->topo_graph->size = 0;

    _topo_sort(z, z->topo_graph);
    _backward(z, partial);
}


double finite_diff(Node *input, Node *target) {
    double val = input->val; // save for restoring
    double eps = 1e-6;

    // right side
    input->val = val + eps;
    forward(target);
    double grad_plus = target->val;

    input->val = val - eps;
    forward(target);
    double grad_minus = target->val;
    
    // restore value
    input->val = val;

    return (grad_plus - grad_minus) / (2 * eps);
}
