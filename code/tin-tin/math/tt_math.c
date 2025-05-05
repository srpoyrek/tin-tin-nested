#include "tt_math.h"
#include "tt_utils.h"
#include <stdlib.h>
#include <string.h>

/* ---------- helpers ---------------------------------------------------- */
int32_t shift_and_round32(int32_t x, uint8_t k)
{
    if (k == 0) return x;
    int32_t off = 1 << (k - 1);
    if (x < 0) off = -off;
    return (x + off) >> k;
}

int8_t upscale_4_3(int8_t x)   { return clip_int8(x + (x >> 2) + (x >> 4)); }
int8_t downscale_4_5(int8_t x) { return clip_int8(x - (x >> 2)); }

uint8_t eff_bitwidth32(int32_t v) { return bitwidth32(v); }

uint8_t eff_bitwidth_array(const int32_t *p, size_t n)
{
    uint32_t maxa = 0;
    for (size_t i = 0; i < n; ++i) {
        uint32_t a = (p[i] < 0) ? -(uint32_t)p[i] : (uint32_t)p[i];
        if (a > maxa) maxa = a;
    }
    return maxa ? 32u - __builtin_clz(maxa) : 1;
}