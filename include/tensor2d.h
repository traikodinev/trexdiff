#ifndef TENSOR2D_H
#define TENSOR2D_H


#include <stdlib.h>
#include <stdbool.h>
#include <cblas.h>


typedef struct Tensor2D {
    double* matrix;
    size_t M;
    size_t N;

    // transposed
    bool T;
} Tensor2D;


// initilization functions
Tensor2D* tensor2d_from_array(size_t M, size_t N, double* matrix);
Tensor2D* tensor2d_zeros(size_t M, size_t N);
Tensor2D* tensor2d_ones(size_t M, size_t N);
void tensor2d_set_zeros(Tensor2D* tensor);

void tensor2d_free(Tensor2D* tensor);


// matrix operations
Tensor2D* tensor2d_matmul(Tensor2D* A, Tensor2D* B);
Tensor2D* tensor2d_transpose(Tensor2D* A);
Tensor2D* tensor2d_add(Tensor2D* A, Tensor2D* B);
Tensor2D* tensor2d_sub(Tensor2D* A, Tensor2D* B);
Tensor2D* tensor2d_scalar_mul(Tensor2D* A, double scalar);

// compute A + scalar * B in-place (A is modified)
void tensor2d_add_inplace(Tensor2D* A, const Tensor2D* B, double scalar);

Tensor2D* tensor2d_scalar_add(Tensor2D* A, double scalar);
Tensor2D* tensor2d_scalar_mul(Tensor2D* A, double scalar);
void tensor2d_scalar_mul_inplace(Tensor2D* A, double scalar);
Tensor2D* tensor2d_sqrt(Tensor2D* A);
Tensor2D* tensor2d_pow(Tensor2D* A, double exponent);

Tensor2D* tensor2d_elwise_mul(Tensor2D* A, const Tensor2D* B);
Tensor2D* tensor2d_elwise_div(Tensor2D* A, const Tensor2D* B);

Tensor2D* tensor2d_relu(Tensor2D* A);
Tensor2D* tensor2d_sigmoid(Tensor2D* A);

Tensor2D* tensor2d_add_broadcast(Tensor2D* A, Tensor2D* B, double scalar); // A + scalar * B

// TODO: inverse solver/determinant

#endif
