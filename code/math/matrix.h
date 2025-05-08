/**
 * @file matrix.h
 * @brief mat multiplication in int32_t with accumulator and other operations
 * @author Shreyas Poyrekar
 * @date May 7, 2025
 * @license MIT
 */
#ifndef _MATRIX_H_
#define _MATRIX_H_
#include "tt_types.h"

static inline void matrix_mul( const tensor_t *A, const tensor_t *X, int32_t *acc_buffer);

#endif // TT_MATMUL_ACC_H
