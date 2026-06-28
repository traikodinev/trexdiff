#include <stdio.h>
#include <math.h>
#include "minunit.h"
#include "tensor2d.h"

int tests_run = 0;

#define EPSILON 1e-4


static char *test_matmul(void) {
    double A_data[] = {1, 2, 3, 4, 5, 6};
    double B_data[] = {7, 8, 9, 10, 11, 12};
    double C_data[] = {58, 64, 139, 154};

    Tensor2D *A = tensor2d_from_array(2, 3, A_data);
    Tensor2D *B = tensor2d_from_array(3, 2, B_data);
    Tensor2D *C = tensor2d_matmul(A, B);

    mu_assert("matmul: NULL result", C != NULL);
    mu_assert("matmul: wrong M", C->M == 2);
    mu_assert("matmul: wrong N", C->N == 2);
    for (size_t i = 0; i < C->M * C->N; ++i)
        mu_assert_close("matmul: value mismatch", C->matrix[i], C_data[i]);

    tensor2d_free(A);
    tensor2d_free(B);
    tensor2d_free(C);
    return 0;
}

static char *test_zeros(void) {
    Tensor2D *A = tensor2d_zeros(3, 4);
    mu_assert("zeros: wrong M", A->M == 3);
    mu_assert("zeros: wrong N", A->N == 4);
    for (size_t i = 0; i < 12; ++i)
        mu_assert_close("zeros: non-zero element", A->matrix[i], 0.0);
    tensor2d_free(A);
    return 0;
}

static char *test_ones(void) {
    Tensor2D *A = tensor2d_ones(2, 5);
    mu_assert("ones: wrong M", A->M == 2);
    mu_assert("ones: wrong N", A->N == 5);
    for (size_t i = 0; i < 10; ++i)
        mu_assert_close("ones: element != 1", A->matrix[i], 1.0);
    tensor2d_free(A);
    return 0;
}

static char *test_from_array(void) {
    double data[] = {1.0, 2.0, 3.0, 4.0};
    Tensor2D *A = tensor2d_from_array(2, 2, data);
    mu_assert("from_array: wrong M", A->M == 2);
    mu_assert("from_array: wrong N", A->N == 2);
    for (size_t i = 0; i < 4; ++i)
        mu_assert_close("from_array: value mismatch", A->matrix[i], data[i]);
    
    // mutating the source array must not affect the tensor (deep copy)
    data[0] = 99.0;
    mu_assert_close("from_array: not a deep copy", A->matrix[0], 1.0);
    tensor2d_free(A);
    return 0;
}

static char *test_copy_inplace(void) {
    double src_data[] = {1.0, 2.0, 3.0, 4.0};
    Tensor2D *src = tensor2d_from_array(2, 2, src_data);
    Tensor2D *dst = tensor2d_zeros(2, 2);
    Tensor2D *wrong_shape = tensor2d_zeros(1, 4);

    mu_assert("copy_inplace: failed", tensor2d_copy_inplace(dst, src) == 0);
    for (size_t i = 0; i < 4; ++i)
        mu_assert_close("copy_inplace: value mismatch", dst->matrix[i], src_data[i]);
    mu_assert("copy_inplace: accepted incompatible shape",
              tensor2d_copy_inplace(dst, wrong_shape) != 0);

    tensor2d_free(src);
    tensor2d_free(dst);
    tensor2d_free(wrong_shape);
    return 0;
}

static char *test_transpose(void) {
    double data[] = {1, 2, 3, 4, 5, 6};
    Tensor2D *A   = tensor2d_from_array(2, 3, data);
    Tensor2D *A_T = tensor2d_transpose(A);

    mu_assert("transpose: wrong M", A_T->M == 3);
    mu_assert("transpose: wrong N", A_T->N == 2);
    for (size_t i = 0; i < A->M; ++i)
        for (size_t j = 0; j < A->N; ++j)
            mu_assert_close("transpose: value mismatch",
                A->matrix[i * A->N + j], A_T->matrix[j * A_T->N + i]);

    tensor2d_free(A);
    tensor2d_free(A_T);
    return 0;
}

static char *test_add(void) {
    double a[]        = {1, 2, 3, 4};
    double b[]        = {5, 6, 7, 8};
    double expected[] = {6, 8, 10, 12};
    Tensor2D *A = tensor2d_from_array(2, 2, a);
    Tensor2D *B = tensor2d_from_array(2, 2, b);
    Tensor2D *C = tensor2d_add(A, B);
    mu_assert("add: NULL result", C != NULL);
    for (size_t i = 0; i < 4; ++i)
        mu_assert_close("add: value mismatch", C->matrix[i], expected[i]);
    tensor2d_free(A);
    tensor2d_free(B);
    tensor2d_free(C);
    return 0;
}

static char *test_sub(void) {
    double a[]        = {5, 6, 7, 8};
    double b[]        = {1, 2, 3, 4};
    double expected[] = {4, 4, 4, 4};
    Tensor2D *A = tensor2d_from_array(2, 2, a);
    Tensor2D *B = tensor2d_from_array(2, 2, b);
    Tensor2D *C = tensor2d_sub(A, B);
    mu_assert("sub: NULL result", C != NULL);
    for (size_t i = 0; i < 4; ++i)
        mu_assert_close("sub: value mismatch", C->matrix[i], expected[i]);
    tensor2d_free(A);
    tensor2d_free(B);
    tensor2d_free(C);
    return 0;
}

static char *test_scalar_mul(void) {
    double data[]     = {1, 2, 3, 4};
    double expected[] = {2.5, 5.0, 7.5, 10.0};
    Tensor2D *A = tensor2d_from_array(2, 2, data);
    Tensor2D *C = tensor2d_scalar_mul(A, 2.5);
    mu_assert("scalar_mul: NULL result", C != NULL);
    for (size_t i = 0; i < 4; ++i)
        mu_assert_close("scalar_mul: value mismatch", C->matrix[i], expected[i]);
    tensor2d_free(A);
    tensor2d_free(C);
    return 0;
}

static char *test_matmul_identity(void) {
    double A_data[] = {1, 2, 3, 4, 5, 6};
    double I_data[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    Tensor2D *A = tensor2d_from_array(2, 3, A_data);
    Tensor2D *Identity = tensor2d_from_array(3, 3, I_data);
    Tensor2D *C = tensor2d_matmul(A, Identity);
    mu_assert("matmul_identity: NULL result", C != NULL);
    mu_assert("matmul_identity: wrong M", C->M == 2);
    mu_assert("matmul_identity: wrong N", C->N == 3);
    for (size_t i = 0; i < 6; ++i)
        mu_assert_close("matmul_identity: value mismatch", C->matrix[i], A_data[i]);
    tensor2d_free(A);
    tensor2d_free(Identity);
    tensor2d_free(C);
    return 0;
}

static char *test_matmul_shape_mismatch(void) {
    Tensor2D *A = tensor2d_zeros(2, 3);
    Tensor2D *B = tensor2d_zeros(4, 2);
    Tensor2D *C = tensor2d_matmul(A, B);
    mu_assert("matmul_shape_mismatch: expected NULL", C == NULL);
    tensor2d_free(A);
    tensor2d_free(B);
    return 0;
}

static char *test_add_shape_mismatch(void) {
    Tensor2D *A = tensor2d_zeros(2, 3);
    Tensor2D *B = tensor2d_zeros(3, 2);
    Tensor2D *C = tensor2d_add(A, B);
    mu_assert("add_shape_mismatch: expected NULL", C == NULL);
    tensor2d_free(A);
    tensor2d_free(B);
    return 0;
}


static char *all_tests(void) {
    mu_run_test(test_matmul);
    mu_run_test(test_zeros);
    mu_run_test(test_ones);
    mu_run_test(test_from_array);
    mu_run_test(test_copy_inplace);
    mu_run_test(test_transpose);
    mu_run_test(test_add);
    mu_run_test(test_sub);
    mu_run_test(test_scalar_mul);
    mu_run_test(test_matmul_identity);
    mu_run_test(test_matmul_shape_mismatch);
    mu_run_test(test_add_shape_mismatch);
    return 0;
}

int main(void) {
    char *result = all_tests();
    if (result != 0) {
        printf("FAILED: %s\n", result);
    } else {
        printf("ALL TESTS PASSED\n");
    }
    printf("Tests run: %d\n", tests_run);
    return result != 0;
}
