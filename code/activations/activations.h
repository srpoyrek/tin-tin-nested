#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <math.h>
#include <stdint.h>


typedef int8_t (*Activation_i8_t)(int8_t x);

typedef float (*Activation_flt_t)(float x);
/*
 * Activation functions for both integer and floating-point use.
 * Static inline for zero overhead.
 */

/* ---------------- Integer activations (int8) --------------------- */

/**
 * ReLU activation: max(0, x)
 */
static inline int8_t relu_i8(int8_t x) {
    return x < 0 ? 0 : x;
}

/**
 * Leaky ReLU with slope 1/4 for x < 0: x >= 0 ? x : x/4
 * Uses arithmetic shift for efficiency.
 */
static inline int8_t leaky_relu_i8(int8_t x) {
    return x < 0 ? (int8_t)(x >> 2) : x;
}

/**
 * Hard tanh: clamps to [-128, 127] (identity here but provided for API symmetry)
 */
static inline int8_t hard_tanh_i8(int8_t x) {
    return x;
}

/**
 * Identity activation (linear)
 */
static inline int8_t linear_i8(int8_t x) {
    return x;
}

/* ---------------- Floating-point activations ---------------------- */

/**
 * ReLU activation: max(0.0, x)
 */
static inline float relu_f(float x) {
    return x < 0.0f ? 0.0f : x;
}

/**
 * Leaky ReLU with slope 0.01
 */
static inline float leaky_relu_f(float x) {
    return x < 0.0f ? 0.01f * x : x;
}

/**
 * Sigmoid activation: 1 / (1 + exp(-x))
 */
static inline float sigmoid_f(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/**
 * Hyperbolic tangent activation
 */
static inline float tanh_f(float x) {
    return tanhf(x);
}

/**
 * Softmax activation over an array in-place.
 * - x: input array
 * - y: output array (can be same as x)
 * - n: length
 */
static inline void softmax_f(const float *x, float *y, int n) {
    /* find max for numeric stability */
    float max_val = x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] > max_val) max_val = x[i];
    }
    /* exponentiate and sum */
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        y[i] = expf(x[i] - max_val);
        sum += y[i];
    }
    /* normalize */
    for (int i = 0; i < n; ++i) {
        y[i] /= sum;
    }
}

#endif // ACTIVATIONS_H
