#ifndef TT_MATH_H
#define TT_MATH_H
#include "tt_types.h"

/* integer helpers ---------------------------------------------------------*/
int32_t shift_and_round32(int32_t x, uint8_t k);   /* divide by 2^k, nearest */
int8_t  upscale_4_3(int8_t x);                     /* x *= 4/3, clip */
int8_t  downscale_4_5(int8_t x);                   /* x *= 4/5, clip */

/* effective bit‑width of an int32_t (magnitude) */
uint8_t eff_bitwidth32(int32_t v);
uint8_t eff_bitwidth_array(const int32_t *p, size_t n);

#endif
