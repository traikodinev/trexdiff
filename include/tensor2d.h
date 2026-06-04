#ifndef TREXDIFF_H
#define TREXDIFF_H


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

void tensor2d_free(Tensor2D* tensor);


// matrix operations
Tensor2D* tensor2d_matmul(Tensor2D* A, Tensor2D* B);
Tensor2D* tensor2d_transpose(Tensor2D* A);
Tensor2D* tensor2d_add(Tensor2D* A, Tensor2D* B);
Tensor2D* tensor2d_sub(Tensor2D* A, Tensor2D* B);
Tensor2D* tensor2d_scalar_mul(Tensor2D* A, double scalar);


// TODO: inverse solver/determinant

#endif
