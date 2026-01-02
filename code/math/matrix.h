/**
 * @file matrix.h
 * @brief Matrix multiplication APIs
 * @author Shreyas Poyrekar
 * @date May 7, 2025
 */

#ifndef MATRIX_H
#define MATRIX_H

#include "tt_types.h"

/**
 * @brief Function pointer for matrix multiplication
 *
 * @param W Weight tensor (flattened 2D matrix)
 * @param X Input tensor (1D vector or batch of vectors)
 * @param acc_buffer Output buffer (int32_t array)
 *
 * Dimensions must be known externally:
 * - W: (OUT × IN) flattened
 * - X: (IN) or (BATCH × IN) flattened
 * - acc_buffer: (OUT) or (BATCH × OUT)
 */
typedef void (*matrix_mul_func)(const tensor_t *W, const tensor_t *X, void *acc_buffer);

/**
 * @brief Matrix-vector multiply: y = W × x
 *
 * Expects:
 * - W->data: int8_t[OUT × IN] flattened row-major
 * - X->data: uint8_t[IN] or int8_t[IN]
 * - acc_buffer: int32_t[OUT]
 *
 * Dimensions: OUT = W->len / X->len
 */
void matrix_mul_1di8(const tensor_t *W, const tensor_t *X, void *acc_buffer);

#endif /* MATRIX_H */
