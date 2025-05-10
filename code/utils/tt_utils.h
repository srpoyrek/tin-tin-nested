#ifndef TT_UTILS_H
#define TT_UTILS_H
#include <stdint.h>

typedef int8_t (*clip_t)(int32_t x);


static inline int8_t clip_int8(int32_t x)
{
    if (x >  INT8_MAX) return  INT8_MAX;
    if (x < INT8_MIN) return INT8_MIN;
    return (int8_t)x;
}

/* effective bit‑width of signed 32‑bit value */
static inline uint8_t bitwidth32(int32_t v)
{
    if (v == 0) return 1;
    uint32_t a = (v < 0) ? -v : v;
    return 32u - __builtin_clz(a);
}

#endif
