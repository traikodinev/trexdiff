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
    node->partial = 0.0;
    node->op_type = OP_NOOP;
    node->visited = false;

    return node;
}


void free_node(Node *n) {
    free(n->inputs);
    if (n->topo_graph) {
        free(n->topo_graph->graph);
        free(n->topo_graph);
    }
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

    return node;
}


Node* transform(Node *a, OpType op_type) {
    Node* node = init(0.0);
    node->input_count = 1;
    node->inputs = malloc(sizeof(Node*));
    node->inputs[0] = a;
    node->op_type = op_type;

    return node;
}


void reset_visited(Node *z) {
    z->visited = false;

    if (z -> input_count == 0)
        return;

    for (unsigned short i = 0; i < z->input_count; ++ i)
        reset_visited(z->inputs[i]);
}


int reset_and_count(Node *z) {
    // TODO: This is not an exact node count due to multiple visits 
    z->visited = false;

    if (z->input_count == 0)
        return 1;

    size_t count = 1;
    for (unsigned short i = 0; i < z->input_count; ++ i)
        count += reset_and_count(z->inputs[i]);

    return count;
}


void zerograd(Node *z) {
    z->grad = 0.0;

    if (z -> input_count == 0)
        return;

    for (unsigned short i = 0; i < z->input_count; ++ i)
        zerograd(z->inputs[i]);
}


static void _forward(Node* z) {
    if (z -> input_count == 0 || z->visited)
        return;

    for (unsigned short i = 0; i < z->input_count; ++ i)
        _forward(z->inputs[i]);
    
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
        case OP_RELU:
            z->val = z->inputs[0]->val < 0 ? 0 : z->inputs[0]->val;
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

    for (int i = start->topo_graph->size - 1; i >= 0; --i)
        start->topo_graph->graph[i]->partial = 0.0;

    
    start->partial = partial;
    for (int i = start->topo_graph->size - 1; i >= 0; --i) {
        z = start->topo_graph->graph[i];

        // gradient accumulation
        z->grad += z->partial;

        switch (z->op_type) {
            case OP_ADD:
                z->inputs[0]->partial += z->partial;
                z->inputs[1]->partial += z->partial;
                break;
            case OP_MUL:
                z->inputs[1]->partial += z->partial * z->inputs[0]->val;
                z->inputs[0]->partial += z->partial * z->inputs[1]->val;
                break;
            case OP_SUB:
                z->inputs[0]->partial += z->partial;
                z->inputs[1]->partial += -z->partial;
                break;
            case OP_RELU:
                z->inputs[0]->partial += z->inputs[0]->val <= 0 ? 0 : z->partial;
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

    if (z->input_count > 0) {
        for (unsigned short i = 0; i < z->input_count; ++ i)
            _topo_sort(z->inputs[i], topo_graph);
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
    reset_visited(target);
    input->visited = true; // stop at input
    _forward(target);
    double grad_plus = target->val;

    input->val = val - eps;
    reset_visited(target);
    input->visited = true; // stop at input
    _forward(target);
    double grad_minus = target->val;

    // restore value
    input->val = val;

    return (grad_plus - grad_minus) / (2 * eps);
}
