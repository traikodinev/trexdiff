#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include "trexdiff.h"


Node* init(Tensor2D* val) {
    Node* node = malloc(sizeof(Node));
    node->input_count = 0;
    node->inputs = NULL;
    node->topo_graph = NULL;
    node->val = val;
    node->grad = tensor2d_zeros(val->M, val->N);
    node->partial = tensor2d_zeros(val->M, val->N);
    node->op_type = OP_NOOP;
    node->visited = false;

    return node;
}


// free only node metadata (inputs array, topo_graph, node struct); tensors are NOT freed
void free_node_shallow(Node *n) {
    free(n->inputs);

    if (n->topo_graph) {
        free(n->topo_graph->graph);
        free(n->topo_graph);
    }
    free(n);
}


void free_node(Node *n) {
    tensor2d_free(n->val);
    tensor2d_free(n->grad);
    tensor2d_free(n->partial);
    free(n->inputs);
    if (n->topo_graph) {
        free(n->topo_graph->graph);
        free(n->topo_graph);
    }
    free(n);
}


Node* combine(Node* a, Node* b, OpType op_type) {

    // todo: this is ugly, we should change combine (?)
    //  work out best API for this
    size_t outM = a->val->M;
    size_t outN = (op_type == OP_MUL) ? b->val->N : a->val->N;
    Node* node = init(tensor2d_zeros(outM, outN));
    
    // adding 2 nodes means 2 inputs
    node->input_count = 2;
    node->inputs = malloc(sizeof(Node*) * 2);

    node->inputs[0] = a;
    node->inputs[1] = b;

    switch (op_type) {
        case OP_ADD:
        case OP_SUB:
            if (a->val->M != b->val->M || a->val->N != b->val->N) {
                fprintf(
                    stderr,
                    "[WARNING] combine: incompatible dims for add/sub, %zux%zu and %zux%zu, transpose not implemented\n",
                    a->val->M, a->val->N, b->val->M, b->val->N
                );
            }

            node->val->M = a->val->M;
            node->val->N = a->val->N;
            node->grad->M = a->val->M;
            node->grad->N = a->val->N;
            node->partial->M = a->val->M;
            node->partial->N = a->val->N;
            break;
        case OP_MUL:
            if (a->val->N != b->val->M) {
                fprintf(
                    stderr,
                    "[WARNING] combine: incompatible dims for mul, %zux%zu and %zux%zu, transpose not implemented\n",
                    a->val->M, a->val->N, b->val->M, b->val->N
                );
            }

            node->val->M = a->val->M;
            node->val->N = b->val->N;
            node->grad->M = a->val->M;
            node->grad->N = b->val->N;
            node->partial->M = a->val->M;
            node->partial->N = b->val->N;
            break;
        case OP_RELU:
        case OP_SIGMOID:
        case OP_LN:
        case OP_SIN:
        case OP_COS:
        case OP_NOOP:
            node->val->M = a->val->M;
            node->val->N = a->val->N;
            node->grad->M = a->val->M;
            node->grad->N = a->val->N;
            node->partial->M = a->val->M;
            node->partial->N = a->val->N;
            break;
        case OP_TRANSPOSE:
            node->val->M = a->val->N;
            node->val->N = a->val->M;
            node->grad->M = a->val->N;
            node->grad->N = a->val->M;
            node->partial->M = a->val->N;
            node->partial->N = a->val->M;
            break;
        case OP_BROADCAST_ADD:
        case OP_BROADCAST_SUB:
            if (a->val->N != b->val->N && b->val->M != 1) {
                fprintf(
                    stderr,
                    "[WARNING] combine: incompatible dims for broadcast, %zux%zu and %zux%zu\n",
                    a->val->M, a->val->N, b->val->M, b->val->N
                );
            }
            node->val->M = a->val->M;
            node->val->N = a->val->N;
            node->grad->M = a->val->M;
            node->grad->N = a->val->N;
            node->partial->M = a->val->M;
            node->partial->N = a->val->N;
            break;
        default:
            break;
    }

    node->op_type = op_type;

    return node;
}


Node* transform(Node *a, OpType op_type) {
    Node* node = init(tensor2d_zeros(a->val->M, a->val->N));
    node->input_count = 1;
    node->inputs = malloc(sizeof(Node*));
    node->inputs[0] = a;
    node->op_type = op_type;

    return node;
}


Node* transpose(Node* a) {
    // TODO: transpose (and functionals) as inplace (no tensor copy)
    Node* node = init(tensor2d_zeros(a->val->N, a->val->M));
    node->input_count = 1;
    node->inputs = malloc(sizeof(Node*));
    node->inputs[0] = a;
    node->op_type = OP_TRANSPOSE;
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
    tensor2d_set_zeros(z->grad);

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
        case OP_ADD: {
            tensor2d_set_zeros(z->val);
            tensor2d_add_inplace(z->val, z->inputs[0]->val, 1.0);
            tensor2d_add_inplace(z->val, z->inputs[1]->val, 1.0);
            break;
        }
        case OP_MUL: {
            Tensor2D *old = z->val;
            // dgemm does not support in-place multiplication
            z->val = tensor2d_matmul(z->inputs[0]->val, z->inputs[1]->val);
            tensor2d_free(old);
            break;
        }
        case OP_SUB: {
            tensor2d_set_zeros(z->val);
            tensor2d_add_inplace(z->val, z->inputs[0]->val, 1.0);
            tensor2d_add_inplace(z->val, z->inputs[1]->val, -1.0);
            break;
        }
        case OP_RELU:
            // z->val = z->inputs[0]->val < 0 ? 0 : z->inputs[0]->val;
            for (size_t i = 0; i < z->inputs[0]->val->M * z->inputs[0]->val->N; ++ i)
                z->val->matrix[i] = z->inputs[0]->val->matrix[i] <= 0 ? 0 : z->inputs[0]->val->matrix[i];
            break;
        case OP_SIGMOID:
            // TODO: use SSE or AVX to optimize this
            // z->val = 1.0 / (1.0 + exp(-z->inputs[0]->val));
            for (size_t i = 0; i < z->inputs[0]->val->M * z->inputs[0]->val->N; ++ i)
                z->val->matrix[i] = 1.0 / (1.0 + exp(-z->inputs[0]->val->matrix[i]));
            break;
        case OP_LN:
            // z->val = log(z->inputs[0]->val);
            for (size_t i = 0; i < z->inputs[0]->val->M * z->inputs[0]->val->N; ++ i)
                z->val->matrix[i] = log(z->inputs[0]->val->matrix[i]);
            break;
        case OP_SIN:
            for (size_t i = 0; i < z->inputs[0]->val->M * z->inputs[0]->val->N; ++ i)
                z->val->matrix[i] = sin(z->inputs[0]->val->matrix[i]);
            break;
        case OP_COS:
            for (size_t i = 0; i < z->inputs[0]->val->M * z->inputs[0]->val->N; ++ i)
                z->val->matrix[i] = cos(z->inputs[0]->val->matrix[i]);
            break;
        case OP_TRANSPOSE:
            // TODO: Better transpose inplace
            for (size_t i = 0; i < z->inputs[0]->val->M; ++i)
                for (size_t j = 0; j < z->inputs[0]->val->N; ++j)
                    z->val->matrix[j * z->val->N + i] = z->inputs[0]->val->matrix[i * z->inputs[0]->val->N + j];
            break;
        case OP_BROADCAST_ADD:
        case OP_BROADCAST_SUB: {
            int sign = (z->op_type == OP_BROADCAST_ADD) ? 1 : -1;
            for (size_t i = 0; i < z->inputs[0]->val->M; ++i)
                for (size_t j = 0; j < z->inputs[0]->val->N; ++j)
                    z->val->matrix[i * z->val->N + j] = z->inputs[0]->val->matrix[i * z->inputs[0]->val->N + j] + sign * z->inputs[1]->val->matrix[j];
            break;
        }
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


static inline void _backward(Node *start, Tensor2D* partial) {
    Node* z;

    for (int i = start->topo_graph->size - 1; i >= 0; --i)
        tensor2d_set_zeros(start->topo_graph->graph[i]->partial);

    // TODO: copy in-place method instead of manual copy
    memcpy(start->partial->matrix, partial->matrix, start->partial->M * start->partial->N * sizeof(double));

    for (int i = start->topo_graph->size - 1; i >= 0; --i) {
        z = start->topo_graph->graph[i];

        // gradient accumulation
        // z->grad += z->partial;
        tensor2d_add_inplace(z->grad, z->partial, 1.0);

        switch (z->op_type) {
            case OP_ADD:
                // z->inputs[0]->partial += z->partial;
                // z->inputs[1]->partial += z->partial;
                tensor2d_add_inplace(z->inputs[0]->partial, z->partial, 1.0);
                tensor2d_add_inplace(z->inputs[1]->partial, z->partial, 1.0);
                break;
            case OP_MUL: {
                // z->inputs[1]->partial += z->partial * z->inputs[0]->val;
                // z->inputs[0]->partial += z->partial * z->inputs[1]->val;
                // 
                // z = A @ B        => NxK = MxN @ NxK
                // dA = dz @ B^T    => MxN = NxK @ KxN
                // dB = A^T @ dz    => NxK = MxN @ NxK
                Tensor2D *BT = tensor2d_transpose(z->inputs[1]->val);
                Tensor2D *AT = tensor2d_transpose(z->inputs[0]->val);
                Tensor2D *dA = tensor2d_matmul(z->partial, BT);
                Tensor2D *dB = tensor2d_matmul(AT, z->partial);
                tensor2d_add_inplace(z->inputs[0]->partial, dA, 1.0);
                tensor2d_add_inplace(z->inputs[1]->partial, dB, 1.0);
                tensor2d_free(BT);
                tensor2d_free(AT);
                tensor2d_free(dA);
                tensor2d_free(dB);
                break;

            }
            case OP_SUB: {
                // z->inputs[0]->partial += z->partial;
                // z->inputs[1]->partial += -z->partial
                tensor2d_add_inplace(z->inputs[0]->partial, z->partial, 1.0);
                tensor2d_add_inplace(z->inputs[1]->partial, z->partial, -1.0);
                break;
            }
            case OP_RELU:
                // z->inputs[0]->partial += z->inputs[0]->val <= 0 ? 0 : z->partial;
                for (size_t i = 0; i < z->inputs[0]->val->M * z->inputs[0]->val->N; ++ i)
                    z->inputs[0]->partial->matrix[i] += z->inputs[0]->val->matrix[i] <= 0 ? 0 : z->partial->matrix[i];
                break;
            case OP_SIGMOID:
                // z->inputs[0]->partial += z->partial * (z->val * (1.0 - z->val));
                for (size_t i = 0; i < z->inputs[0]->val->M * z->inputs[0]->val->N; ++ i)
                    z->inputs[0]->partial->matrix[i] += z->partial->matrix[i] * (z->val->matrix[i] * (1.0 - z->val->matrix[i]));
                break;
            case OP_LN:
                // z->inputs[0]->partial += z->partial / z->inputs[0]->val;
                for (size_t i = 0; i < z->inputs[0]->val->M * z->inputs[0]->val->N; ++ i)
                    z->inputs[0]->partial->matrix[i] += z->partial->matrix[i] / z->inputs[0]->val->matrix[i];
                break;
            case OP_SIN:
                // z->inputs[0]->partial += z->partial * cos(z->inputs[0]->val);
                for (size_t i = 0; i < z->inputs[0]->val->M * z->inputs[0]->val->N; ++ i)
                    z->inputs[0]->partial->matrix[i] += z->partial->matrix[i] * cos(z->inputs[0]->val->matrix[i]);
                break;
            case OP_COS:
                // z->inputs[0]->partial += -z->partial * sin(z->inputs[0]->val);
                for (size_t i = 0; i < z->inputs[0]->val->M * z->inputs[0]->val->N; ++ i)
                    z->inputs[0]->partial->matrix[i] += -z->partial->matrix[i] * sin(z->inputs[0]->val->matrix[i]);
                break;
            case OP_TRANSPOSE:
                // transparent routing of partials, since d(A^T) = (dA)^T
                for (size_t i = 0; i < z->inputs[0]->val->M; ++i)
                    for (size_t j = 0; j < z->inputs[0]->val->N; ++j)
                        z->inputs[0]->partial->matrix[i * z->inputs[0]->partial->N + j] += z->partial->matrix[j * z->partial->N + i];
                break;
            case OP_BROADCAST_ADD:
            case OP_BROADCAST_SUB:
                // sum of partials across broadcast dimension
                tensor2d_add_inplace(z->inputs[0]->partial, z->partial, 1.0);
                int sign = (z->op_type == OP_BROADCAST_ADD) ? 1 : -1;
                for (size_t j = 0; j < z->inputs[1]->val->N; ++j) {
                    double col_sum = 0.0;
                    for (size_t i = 0; i < z->partial->M; ++i)
                        col_sum += z->partial->matrix[i * z->partial->N + j];
                    z->inputs[1]->partial->matrix[j] += sign * col_sum;
                }
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


int backward(Node *z, Tensor2D* partial) {
    // check that node is 1x1 (single loss gradient)
    if (z->val->M != 1 || z->val->N != 1) {
        // TODO: better error handling
        fprintf(stderr, "[WARNING] backward: target node must be 1x1, not %zux%zu\n", z->val->M, z->val->N);
        return -1;
    }


    if (z->topo_graph) {
        _backward(z, partial);
        return 0;
    }

    // TODO: We can optimize this by allowing calls to e.g. `compile`, a la Pytorch
    //  instead of manually computing the topological sorting
    size_t n_nodes = reset_and_count(z);
    z->topo_graph = malloc(sizeof(NodeArray));
    z->topo_graph->graph = malloc(sizeof(Node*) * n_nodes);
    z->topo_graph->size = 0;

    _topo_sort(z, z->topo_graph);
    _backward(z, partial);

    return 0;
}


Tensor2D* finite_diff(Node *input, Node *target) {
    Tensor2D* val = input->val;
    double eps = 1e-6;
    size_t n = val->M * val->N;
    Tensor2D* grad = tensor2d_zeros(val->M, val->N);

    // note: target must be 1x1, no vector-values Jacobians
    if (target->val->M != 1 || target->val->N != 1) {
        // stderr warning
        fprintf(stderr, "[WARNING] finite_diff: target node must be 1x1, not %zux%zu\n", target->val->M, target->val->N);
        return NULL;
    }

    for (size_t k = 0; k < n; ++k) {
        double orig = val->matrix[k];

        // f(x + eps)
        val->matrix[k] = orig + eps;
        reset_visited(target);
        input->visited = true;
        _forward(target);
        double f_plus = target->val->matrix[0];

        // f(x - eps)
        val->matrix[k] = orig - eps;
        reset_visited(target);
        input->visited = true;
        _forward(target);
        double f_minus = target->val->matrix[0];

        grad->matrix[k] = (f_plus - f_minus) / (2.0 * eps);

        val->matrix[k] = orig;
    }

    return grad;
}
