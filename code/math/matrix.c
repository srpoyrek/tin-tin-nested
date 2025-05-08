#include "matrix.h"
/**
 * @file matrix.h
 * @brief mat multiplication in int32_t with accumulator and other operations
 * @author Shreyas Poyrekar
 * @date May 7, 2025
 * @license MIT
 */

/**
 * @brief matrix mulitplication of tensors
 * @param A pointer of Activation Tensor
 * @param X pointer of Weight Tensor
 * @param acc_buffer int32_t pointer accumulator to perfom the operations
 * @return NULL
 */
static inline void matrix_mul( const tensor_t *A, const tensor_t *X, int32_t *acc_buffer) {
    if (!A || !X || !acc_buffer) return;
    size_t OUT = A->len / X->len;
    size_t IN  = X->len;
    for (size_t r = 0; r < OUT; ++r) {
        int32_t sum = 0;
        const int8_t *row = &A->data[r * IN];
        for (size_t c = 0; c < IN; ++c) {
            sum += (int32_t)row[c] * (int32_t)X->data[c];
        }
        acc_buffer[r] = sum;
    }
    return;
}