/**
 * @file scale.h
 * @brief Scale tracking for integer-based neural network training
 * @author Shreyas Poyrekar
 * @date May 7, 2025
 * @license MIT
 */

#ifndef SCALE_H
#define SCALE_H

#include <stdint.h>

/**
 * @brief Basic scale structure
 * Formula: real value = Q (int8) * 2^S × (4/5)^U × (3/4)^D
 */
typedef struct {
    int8_t S;   /**< Power-of-two shifts: 2^S */
    int8_t U;   /**< Up-scales:  (4/5)^U */
    int8_t D;   /**< Down-scales:  (3/4)^D */
} scale_t;

/* ==================== Scale Operations ==================== */

/**
 * @brief Initialize scale to 1.0 (all exponents zero)
 */
static void scale_init(scale_t *s);

/**
 * @brief Combine two scales (for multiplication)
 * Result: S = S_a + S_b, U = U_a + U_b, D = D_a + D_b
 */
void scale_combine(scale_t *dst, const scale_t *a, const scale_t *b);

/**
 * @brief Apply bit shift to scale
 * Positive k: integer >>= k, scale *= 2^k
 * Negative k: integer <<= |k|, scale /= 2^|k|
 */
void scale_shift(scale_t *h, int8_t k);

/**
 * @brief Apply upscale (scale × 4/5)
 * Called when integer is multiplied by 5/4
 */
void scale_up(scale_t *h);

/**
 * @brief Apply downscale (scale × 3/4)
 * Called when integer is multiplied by 4/3
 */
void scale_down(scale_t *h);

/**
 * @brief Copy scale from src to dst
 */
void scale_copy(scale_t *dst, const scale_t *src);

#endif /* SCALE_H */
