#ifndef TT_MATH_H
#define TT_MATH_H
#include "tt_types.h"
#include<limits.h>
/**
 * @typedef scale_by_t
 * @brief Function pointer type for scaling an 8-bit value.
 *
 * @param x  The input int8_t value to be scaled.
 * @return   The scaled int8_t result.
 */
typedef int8_t (*scale_by_t)(int8_t x);

/**
 * @typedef shift_round_t
 * @brief Function pointer type for shifting and rounding a 32-bit value.
 *
 * @param x  The input int32_t value to shift.
 * @param k  Number of bits to shift (0–31).
 * @return   The shifted and rounded int32_t result.
 */
typedef int32_t (*shift_round_t)(int32_t x, uint8_t k);

/**
 * @typedef clip_t
 * @brief Function pointer type for clipping a 32-bit value into an 8-bit range.
 *
 * @param x  The input int32_t value to clip.
 * @return   The clipped int8_t result.
 */
typedef int8_t (*clip_t)(int32_t x);

/**
 * @brief  Clip a 32-bit integer to the signed 8-bit range.
 *
 * If the input exceeds INT8_MAX or is below INT8_MIN, the value
 * is saturated to INT8_MAX or INT8_MIN respectively.
 *
 * @param[in]  x  The int32_t value to clip.
 * @return        The clipped int8_t value.
 */
inline int8_t clip_int8(int32_t x) {
    if (x >  INT8_MAX) return  INT8_MAX;
    if (x < INT8_MIN) return INT8_MIN;
    return (int8_t)x;
}

/**
 * @brief  Arithmetic right-shift with rounding.
 *
 * Performs (x + rounding_offset) >> k, where rounding_offset is
 * 1 << (k-1). For negative x, the offset is negated to achieve
 * symmetric rounding.
 *
 * @param[in]  x  The int32_t value to shift.
 * @param[in]  k  Number of bits to shift (0–31). If zero, no shift is performed.
 * @return        The shifted and rounded int32_t result.
 */
inline int32_t shift_and_round32(int32_t x, uint8_t k) {
    if (k == 0) return x;
    int32_t off = 1 << (k - 1);
    if (x < 0) off = -off;
    return (x + off) >> k;
}

/**
 * @brief  Upscale an 8-bit value by a factor of 4/3, with clipping.
 *
 * Computes x + x/4 + x/16, then clips the result to the int8_t range.
 *
 * @param[in]  x  The int8_t value to upscale.
 * @return        The upscaled and clipped int8_t result.
 */
inline int8_t upscale_4_3(int8_t x) {
    return clip_int8(x + (x >> 2) + (x >> 4));
}

/**
 * @brief  Downscale an 8-bit value by a factor of 4/5, with clipping.
 *
 * Computes x - x/4, then clips the result to the int8_t range.
 *
 * @param[in]  x  The int8_t value to downscale.
 * @return        The downscaled and clipped int8_t result.
 */
inline int8_t downscale_4_5(int8_t x) {
    return clip_int8(x - (x >> 2));
}

/**
 * @brief  Compute the number of bits needed to represent
 *         the largest absolute value in a signed-32 array and then
 *          returns the effective bits to shift to get the MSB.
 * @param[in] p  Pointer to the first element of the int32_t array.
 * @param[in] n  Number of elements in the array.
 * @return        Effective bits to shift.
 */
inline uint8_t eff_shift_bitwidth_array(const int32_t *p, size_t n) {   
    uint8_t width = 0;
    uint32_t maxa = 0;
    for (size_t i = 0; i < n; ++i) {
        maxa = max(abs(p[i]), maxa);
    }
    while (maxa) {
        maxa >>= 1;
        ++width;
    }
    return (width - CHAR_BIT) & -(width > CHAR_BIT);
}

#endif
