#include <string.h>
#include <stdio.h>
#include <math.h>
#include "tensor2d.h"

// initilization functions

static inline Tensor2D* _init_tensor2d(size_t M, size_t N) {
    Tensor2D* tensor = malloc(sizeof(Tensor2D));
    tensor->M = M;
    tensor->N = N;
    tensor->T = false;
    tensor->matrix = malloc(M * N * sizeof(double));
    return tensor;
}


Tensor2D* tensor2d_from_array(size_t M, size_t N, double* matrix) {
    Tensor2D* tensor = _init_tensor2d(M, N);
    memcpy(tensor->matrix, matrix, M * N * sizeof(double));
    return tensor;
}


Tensor2D* tensor2d_zeros(size_t M, size_t N) {
    Tensor2D* tensor = _init_tensor2d(M, N);
    memset(tensor->matrix, 0, M * N * sizeof(double));
    return tensor;
}


void tensor2d_set_zeros(Tensor2D* tensor) {
    memset(tensor->matrix, 0, tensor->M * tensor->N * sizeof(double));
}


Tensor2D* tensor2d_ones(size_t M, size_t N) {
    Tensor2D* tensor = _init_tensor2d(M, N);
    
    for (size_t i = 0; i < M * N; ++ i)
        tensor->matrix[i] = 1.0;

    return tensor;
}


void tensor2d_free(Tensor2D* tensor) {
    free(tensor->matrix);
    free(tensor);
}


// matrix operations
Tensor2D* tensor2d_matmul(Tensor2D* A, Tensor2D* B) {
    if (A->N != B->M)
        return NULL;

    Tensor2D* C = _init_tensor2d(A->M, B->N);
    cblas_dgemm(
        CblasRowMajor,
        A->T ? CblasTrans : CblasNoTrans,
        B->T ? CblasTrans : CblasNoTrans,
        A->M, B->N, A->N, 1.0, A->matrix, A->N, B->matrix, B->N, 0.0, C->matrix, C->N
    );
    return C;
}


Tensor2D* tensor2d_transpose(Tensor2D* A) {
    Tensor2D* T = _init_tensor2d(A->N, A->M);

    for (size_t i = 0; i < A->M; ++i)
        for (size_t j = 0; j < A->N; ++j)
            T->matrix[j * T->N + i] = A->matrix[i * A->N + j];

    return T;
}


static inline Tensor2D* _tensor2d_affine(double a, Tensor2D* A, double b, Tensor2D* B) {
    // TODO: handle transpose
    
    // returna a * A + b * B
    if (A->M != B->M || A->N != B->N) {
        // warn - transpose not implemented
        fprintf(
            stderr,
            "[WARNING] _tensor2d_affine: incompatible dims, %zux%zu and %zux%zu, transpose not implemented\n",
            A->M, A->N, B->M, B->N
        );
        return NULL;
    }

    Tensor2D* C = _init_tensor2d(A->M, A->N);

    for (size_t i = 0; i < A->M * A->N; ++ i)
        C->matrix[i] = a * A->matrix[i] + b * B->matrix[i];

    return C;
}


Tensor2D* tensor2d_add(Tensor2D* A, Tensor2D* B) {
    return _tensor2d_affine(1.0, A, 1.0, B);
}


Tensor2D* tensor2d_sub(Tensor2D* A, Tensor2D* B) {
    return _tensor2d_affine(1.0, A, -1.0, B);
}


Tensor2D* tensor2d_scalar_mul(Tensor2D* A, double scalar) {
    Tensor2D* C = _init_tensor2d(A->M, A->N);

    for (size_t i = 0; i < A->M * A->N; ++ i)
        C->matrix[i] = scalar * A->matrix[i];

    return C;
}

void tensor2d_scalar_mul_inplace(Tensor2D* A, double scalar) {
    for (size_t i = 0; i < A->M * A->N; ++ i)
        A->matrix[i] *= scalar;
}

void tensor2d_scalar_add_inplace(Tensor2D* A, double scalar, int sign) {
    for (size_t i = 0; i < A->M * A->N; ++ i)
        A->matrix[i] += sign * scalar;
}

// inplace operations
void tensor2d_add_inplace(Tensor2D* A, const Tensor2D* B, double scalar) {
    if (A->M != B->M || A->N != B->N) {
        fprintf(
            stderr,
            "[WARNING] tensor2d_add_inplace: incompatible dims, %zux%zu and %zux%zu, transpose not implemented\n",
            A->M, A->N, B->M, B->N
        );
        return;
    }
    for (size_t i = 0; i < A->M * A->N; ++i)
        A->matrix[i] += scalar * B->matrix[i];
}

Tensor2D* tensor2d_scalar_add(Tensor2D* A, double scalar) {
    Tensor2D* C = _init_tensor2d(A->M, A->N);

    for (size_t i = 0; i < A->M * A->N; ++ i)
        C->matrix[i] = scalar + A->matrix[i];

    return C;
}


Tensor2D* tensor2d_sqrt(Tensor2D* A) {
    Tensor2D* C = _init_tensor2d(A->M, A->N);

    for (size_t i = 0; i < A->M * A->N; ++ i)
        C->matrix[i] = sqrt(A->matrix[i]);

    return C;
}


Tensor2D* tensor2d_pow(Tensor2D* A, double exponent) {
    Tensor2D* C = _init_tensor2d(A->M, A->N);

    for (size_t i = 0; i < A->M * A->N; ++ i)
        C->matrix[i] = pow(A->matrix[i], exponent);

    return C;
}

Tensor2D* tensor2d_sin(Tensor2D* A) {
    Tensor2D* C = _init_tensor2d(A->M, A->N);
    for (size_t i = 0; i < A->M * A->N; ++i)
        C->matrix[i] = sin(A->matrix[i]);
    return C;
}

Tensor2D* tensor2d_cos(Tensor2D* A) {
    Tensor2D* C = _init_tensor2d(A->M, A->N);
    for (size_t i = 0; i < A->M * A->N; ++i)
        C->matrix[i] = cos(A->matrix[i]);
    return C;
}


Tensor2D* tensor2d_elwise_mul(Tensor2D* A, const Tensor2D* B) {
    if (A->M != B->M || A->N != B->N) {
        fprintf(
            stderr,
            "[WARNING] tensor2d_elwise_mul: incompatible dims, %zux%zu and %zux%zu, transpose not implemented\n",
            A->M, A->N, B->M, B->N
        );
        return NULL;
    }
    Tensor2D* C = _init_tensor2d(A->M, A->N);
    for (size_t i = 0; i < A->M * A->N; ++i)
        C->matrix[i] = A->matrix[i] * B->matrix[i];
    return C;
}

Tensor2D* tensor2d_elwise_div(Tensor2D* A, const Tensor2D* B) {
    if (A->M != B->M || A->N != B->N) {
        fprintf(
            stderr,
            "[WARNING] tensor2d_elwise_div: incompatible dims, %zux%zu and %zux%zu, transpose not implemented\n",
            A->M, A->N, B->M, B->N
        );
        return NULL;
    }
    Tensor2D* C = _init_tensor2d(A->M, A->N);
    for (size_t i = 0; i < A->M * A->N; ++i)
        C->matrix[i] = A->matrix[i] / B->matrix[i];
    return C;
}


Tensor2D* tensor2d_relu(Tensor2D* A) {
    Tensor2D* C = _init_tensor2d(A->M, A->N);
    for (size_t i = 0; i < A->M * A->N; ++i)
        C->matrix[i] = fmax(0.0, A->matrix[i]);
    return C;
}

Tensor2D* tensor2d_sigmoid(Tensor2D* A) {
    Tensor2D* C = _init_tensor2d(A->M, A->N);
    for (size_t i = 0; i < A->M * A->N; ++i)
        C->matrix[i] = 1.0 / (1.0 + exp(-A->matrix[i]));
    return C;
}


Tensor2D* tensor2d_add_broadcast(Tensor2D* A, Tensor2D* B, double scalar) {
    if (A->N != B->N && B->M != 1) {
        fprintf(
            stderr,
            "[WARNING] tensor2d_add_broadcast: incompatible dims, %zux%zu and %zux%zu, transpose not implemented\n",
            A->M, A->N, B->M, B->N
        );
        return NULL;
    }
    Tensor2D* C = _init_tensor2d(A->M, A->N);
    for (size_t i = 0; i < A->M; ++i) {
        for (size_t j = 0; j < A->N; ++j) {
            C->matrix[i * C->N + j] = A->matrix[i * A->N + j] + scalar * B->matrix[j];
        }
    }
    return C;
}

