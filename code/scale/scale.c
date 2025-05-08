/**
 * @file scale.c
 * @brief contains the source apis for scale operations and definitions
 * @details this file contains scale structure, scale up / down count,
 * roll up, combine scales and scale shift
 * @author Shreyas Poyrekar
 * @date May 7, 2025
 * @license MIT
 */
#include "scale.h"

/**
 * @brief combine scales
 * @param dst pointer of destination
 * @param a pointer of scale one to combine
 * @param b pointer of scale two to combine
 * @return NULL
 */
static inline void scale_combine(scale_t *dst, const scale_t *a, const scale_t *b) {
    if(!a || !b || !dst) return;
#ifdef TENSOR_USE_NESTED
    dst->g.S = a->g.S + b->g.S;
    dst->g.U = a->g.U + b->g.U;
    dst->g.D = a->g.D + b->g.D;
    dst->l.S = a->l.S + b->l.S;
    dst->l.U = a->l.U + b->l.U;
    dst->l.D = a->l.D + b->l.D;
#else
    dst->S = a->S + b->S;
    dst->U = a->U + b->U;
    dst->D = a->D + b->D;
#endif
    return;
}

/**
 * @brief shifts scales
 * @param h pointer of the scale to shift
 * @param k shift by
 * @return NULL
 */
static inline void scale_shift(scale_t *h, int8_t k) {
    if(!h) return;
#ifdef TENSOR_USE_NESTED
    h->l.S += k;
#else
    h->S   += k;
#endif
    return;
}

/**
 * @brief scales up
 * increments by one
 * @param h pointer of the scale to up
 * @return NULL
 */
static inline void scale_up(scale_t *h) {
    if(!h) return;
#ifdef TENSOR_USE_NESTED
    h->l.U += 1;
#else
    h->U   += 1;
#endif
    return;
}

/**
 * @brief scales down
 * decrements by one
 * @param h pointer of the scale to down
 * @return NULL
 */
static inline void scale_down(scale_t *h) {
    if(!h) return;
#ifdef TENSOR_USE_NESTED
    h->l.D += 1;
#else
    h->D += 1;
#endif
    return;
}

#ifdef TENSOR_USE_NESTED
/**
 * @brief rolls up scale
 * @param h pointer of the scale to roll upd
 * @return NULL
 */
static inline void scale_rollup(scale_t *h) {
    if(!h) return;
    const int8_t LIM  = 16, STEP = 8;
    if      (h->l.S >  LIM) { h->g.S += STEP; h->l.S -= STEP; }
    else if (h->l.S < -LIM) { h->g.S -= STEP; h->l.S += STEP; }
    if      (h->l.U >  LIM) { h->g.U += STEP; h->l.U -= STEP; }
    if      (h->l.D >  LIM) { h->g.D += STEP; h->l.D -= STEP; }
}
#endif
