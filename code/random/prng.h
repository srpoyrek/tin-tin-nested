#ifndef PRNG_H
#define PRNG_H

#include <stdint.h>

/* Initialize the PRNG state.  
   If seed==0 you can choose a non‐zero default. */
static inline void prng_init(uint32_t *state, uint32_t seed) {
    *state = seed ? seed : 0xDEADBEEF;
}

/* xorshift32: period ~2³²‑1, very fast on Cortex‑M */
static inline uint32_t prng_next(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return *state = x;
}

/* Uniform int8 in the full [-128,127] range */
static inline int8_t prng_rand_int8(uint32_t *state) {
    /* take the top 8 bits of the 32‑bit state */
    return (int8_t)(prng_next(state) >> 24);
}

/* Uniform in a custom signed range [min,max] */
static inline int8_t prng_rand_range(uint32_t *state, int8_t min, int8_t max) {
    uint32_t r = prng_next(state);
    uint8_t span = (uint8_t)(max - min + 1);
    return (int8_t)(min + (r % span));
}
#endif