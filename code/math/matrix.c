/**
 * @file matrix.c
 * @author Shreyas Poyrekar
 * @date May 7, 2025
 */

#include "matrix.h"

void matrix_mul_1di8(const tensor_t *W, const tensor_t *X, void *acc_buffer) {
    if (!W || !X || !acc_buffer) return;
    if (W->len == 0 || X->len == 0) return;

    // Infer dimensions
    size_t IN = X->len;
    size_t OUT = W->len / IN;

    int8_t *w_data = (int8_t *)W->data;
    int8_t *x_data = (int8_t *)X->data;  // Works for uint8 too (same bit pattern)
    int32_t *acc = (int32_t *)acc_buffer;

    // Matrix-vector multiply
    for (size_t out = 0; out < OUT; out++) {
        int32_t sum = 0;
        const int8_t *row = &w_data[out * IN];

        for (size_t in = 0; in < IN; in++) {
            sum += row[in] * x_data[in];
        }

        acc[out] = sum;
    }
}
