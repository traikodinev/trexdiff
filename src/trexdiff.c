#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "trexdiff.h"


Node* init(double val) {
    Node* node = malloc(sizeof(Node));
    node->input_count = 0;
    node->inputs = NULL;
    node->val = val;
    node->grad = 0.0;
    return node;
}


void free_node(Node *n) {
    free(n->inputs);
    free(n);
}

Node* combine(Node* a, Node* b, OpType op_type) {
    Node* node = malloc(sizeof(Node));
    
    // adding 2 nodes means 2 inputs
    node->input_count = 2;
    node->inputs = malloc(sizeof(Node) * 2);

    node->inputs[0] = a;
    node->inputs[1] = b;

    node->op_type = op_type;
    
    return node;
}

void zerograd(Node *z) {
    z->grad = 0.0;

    if (z -> input_count == 0)
        return;

    zerograd(z->inputs[0]);
    zerograd(z->inputs[1]);
}


void forward(Node *z) {
    if (z -> input_count == 0)
        return;

    forward(z->inputs[0]);
    forward(z->inputs[1]);
    
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
}


void backward(Node *z, double partial) {
    z->grad += partial;

    if (z->input_count == 0)
        return;

    // TODO: toplogical sort instead of recursion
    switch (z->op_type) {
        case OP_ADD:
            backward(z->inputs[0], partial);
            backward(z->inputs[1], partial);
            break;
        case OP_MUL:
            backward(z->inputs[1], partial * z->inputs[0]->val);
            backward(z->inputs[0], partial * z->inputs[1]->val);
            break;
        case OP_SUB:
            backward(z->inputs[0], partial);
            backward(z->inputs[1], -partial);
            break;
        case OP_NOOP:
        default:
            // no-op
            break;
    }
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
