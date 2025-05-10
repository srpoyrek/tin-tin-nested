#include "tt_dense.h"
#include "tt_math.h"
#include "matrix.h"
#include "tt_utils.h"
#include "activations.h"
#include <stdlib.h>
#include <string.h>

/* constants */
#define LR_SHIFT 8    /* lr = 1 / 256   */
#define MARGIN   2    /* Algorithm‑3 line 10 */

#define T_LOW   (2 ^ (CHAR_BIT - 1) / 4)
#define T_HIGH  ((2 ^ (CHAR_BIT - 1) * 7) / 8)


static void align_scale(tensor_t *W, tensor_t *G_buffer);
static inline void s_activation_func(int32_t * acc_buffer, size_t len, Activation_i8_t func);
static inline void s_scale_by_func(tensor_t * Y, size_t len, scale_by_t func);
static inline void s_shift_and_round_func(tensor_t * Y, size_t len, shift_round_t func);
static inline void s_clip_func(tensor_t * Y, size_t len, clip_t func);
static inline int8_t s_max_abs_value(tensor_t * Y, size_t len);

/**
 * @brief Performs a forward pass of a dense (fully connected) layer for tin-tin and tin-tin nested.
 *
 * This function implements the forward computation for a dense layer, including
 * matrix multiplication, activation function application, scaling, shifting,
 * rounding, clipping, and dynamic range adjustment based on the output values.
 *
 * @param W Pointer to the weight tensor. It is assumed to be a 2D tensor
 * where the number of rows corresponds to the output features and the
 * number of columns corresponds to the input features. Must not be NULL.
 * @param X Pointer to the input tensor. It is assumed to be a 1D tensor
 * representing the input features. Must not be NULL.
 * @param Y Pointer to the output tensor. This tensor will store the result
 * of the forward pass. Its length should match the number of output
 * features. Must not be NULL.
 * @param acc_buffer Pointer to an int32_t buffer used for intermediate
 * accumulation during the matrix multiplication. Its size should be at
 * least the length of the output tensor `Y`. Must not be NULL.
 */
void tt_dense_forward(const tensor_t *W, const tensor_t *X, tensor_t *Y, int32_t * acc_buffer, size_t acc_size) {
    if(!W || !X || !Y || !acc_buffer || acc_size != Y->len) return;
    
    // raw int32 matrix-vector multiplication
    matrix_mul(W, X, acc_buffer);

    // apply ReLU to the int32 accumulators
    s_activation_func(acc_buffer, Y->len, relu_i8);

    // get effective max bit width
    uint8_t bw   = eff_bitwidth_array(acc_buffer, Y->len);
    
    // get the bits to shift
    uint8_t ksh  = (bw - CHAR_BIT) & -(bw > CHAR_BIT);

    // shift and round by 32
    s_shift_and_round_func(Y, Y->len, shift_and_round32, ksh);

    // clip by at int8_t
    s_clip_func(Y, Y->len, clip_int8);

    // get the max abs output value in the layer 
    int8_t maxv = s_max_abs_value(Y, Y->len);

    // combine the scales of weights and Activations 
    scale_combine(&Y->s, &W->s, &X->s);

    // shift the scale of Y
    scale_shift(&Y->s, -(int8_t)ksh);

    if(maxv < T_LOW) {
        s_scale_by_func(Y, Y->len, upscale_4_3);
        scale_up(&Y->s);           
    } else if(maxv > T_HIGH) {
        s_scale_by_func(Y, Y->len, downscale_4_5);
        scale_down(&Y->s);  
    }

#ifdef TENSOR_USE_NESTED
    // roll up scale
    scale_rollup(&Y->s);
#endif
    return;
}

/**
 * @brief Applies an activation function to each element of an integer array.
 *
 * This function iterates through an array of 32-bit integers and applies a provided
 * activation function to each element in place.
 *
 * @param acc Pointer to the beginning of the integer array. If NULL, the function returns immediately.
 * @param len The number of elements in the array.
 * @param func A function pointer to the activation function. This function should
 * take an `int32_t` as input and return an `int32_t`. The type
 * `Activation_i8_t` is assumed to be defined as `int32_t (*)(int32_t)`.
 */
static inline void s_activation_func(int32_t *acc, size_t len, Activation_i8_t func) {
    if(!acc) return;
    for (size_t i = 0; i < len; ++i) {
        acc[i] = func(acc[i]);
    }
    return;
}

/**
 * @brief Scales each element of a tensor's data using a provided scaling function.
 *
 * This function iterates through the data array of a given tensor and applies a
 * scaling function to each element in place.
 *
 * @param Y Pointer to the tensor structure. If NULL, the function returns immediately.
 * The tensor structure is assumed to have a data member accessible as `Y->data`.
 * @param len The number of elements in the tensor's data array.
 * @param func A function pointer to the scaling function. This function should
 * take an element from the tensor's data array as input and return
 * the scaled value of the same type. The type `scale_by_t` is assumed
 * to be defined appropriately for the data type of the tensor.
 */
static inline void s_scale_by_func(tensor_t * Y, size_t len, scale_by_t func) {
    if(!Y) return;
    for (size_t i = 0; i < len; ++i) {
        Y->data[i] = func(Y->data[i]);
    }
    return;
}

/**
 * @brief Applies a clipping function to each element of a tensor's data.
 *
 * This function iterates through the data array of a given tensor and applies a
 * clipping function to each element in place. Clipping typically limits values
 * within a specific range.
 *
 * @param Y Pointer to the tensor structure. If NULL, the function returns immediately.
 * The tensor structure is assumed to have a data member accessible as `Y->data`.
 * @param len The number of elements in the tensor's data array.
 * @param func A function pointer to the clipping function. This function should
 * take an element from the tensor's data array as input and return
 * the clipped value of the same type. The type `clip_t` is assumed
 * to be defined appropriately for the data type of the tensor.
 */
static inline void s_clip_func(tensor_t * Y, size_t len, clip_t func) {
    if(!Y) return;
    for (size_t i = 0; i < len; ++i) {
        Y->data[i] = func(Y->data[i]);
    }
    return;
}

/**
 * @brief Applies a shift and rounding operation to each element of a tensor's data,
 * casting the result to an 8-bit signed integer.
 *
 * This function iterates through the data array of a given tensor, applies a
 * shift and rounding function to each element along with a shift parameter, and
 * then casts the result to an `int8_t` before updating the element in place.
 *
 * @param Y Pointer to the tensor structure. If NULL, the function returns immediately.
 * The tensor structure is assumed to have a data member accessible as `Y->data`,
 * which is expected to store `int8_t` values after the operation.
 * @param len The number of elements in the tensor's data array.
 * @param func A function pointer to the shift and rounding function. This function
 * should take an element from the tensor's data array and a `uint8_t`
 * value as input, and return a value that can be cast to `int8_t`.
 * The type `shift_round_t` is assumed to be defined appropriately.
 * @param ksh An unsigned 8-bit integer value used as a shift parameter in the
 * `func` function.
 */
static inline void s_shift_and_round_func(tensor_t * Y, size_t len, shift_round_t func, uint8_t ksh) {
    if(!Y) return;
    for (size_t i = 0; i < len; ++i) {
        Y->data[i] = (int8_t)(func(Y->data[i], ksh));
    }
    return;
}

/**
 * @brief Finds the maximum absolute value among the elements of a tensor's data.
 *
 * This function iterates through the data array of a given tensor and determines
 * the largest absolute value present.
 *
 * @param Y Pointer to the tensor structure. If NULL, the function returns.
 * The tensor structure is assumed to have a data member accessible as `Y->data`.
 * @param len The number of elements in the tensor's data array.
 * @return The maximum absolute value found in the tensor's data, as an 8-bit
 * signed integer (`int8_t`). If the input tensor pointer `Y` is NULL,
 * the behavior is undefined (it currently returns without a value).
 * It is recommended to handle the NULL case by returning a default value
 * like 0.
 */
static inline int8_t s_max_abs_value(tensor_t * Y, size_t len) {
    if(!Y) return;
    int8_t maxv = 0;
    for (size_t i = 0; i < len; ++i) {
        maxv = max(abs(Y->data[i]), maxv);
    }
    return;
}

/* single‑header alignment (Alg 3 lines 1–6) */
static void align_scale(tensor_t *W, tensor_t *G_buffer)
{
    int8_t dS = W->s.S - G_buffer->s.S;
    int8_t dU = W->s.U - G_buffer->s.U;
    int8_t dD = W->s.D - G_buffer->s.D;

    if (dS > 0) {
        for (size_t i = 0; i < G_buffer->len; ++i)
            G_buffer->data[i] <<= dS;
        G_buffer->s.S += dS;
    } else if (dS < 0) {
        dS = -dS;
        for (size_t i = 0; i < G_buffer->len; ++i)
            G_buffer->data[i] = shift_and_round32(G_buffer->data[i], dS);
        G_buffer->s.S -= dS;
    }

    while (dU-- > 0) {
        for (size_t i = 0; i < G_buffer->len; ++i)
            G_buffer->data[i] = upscale_4_3(G_buffer->data[i]);
        G_buffer->s.U++;
    }
    while (dD-- > 0) {
        for (size_t i = 0; i < G_buffer->len; ++i)
            G_buffer->data[i] = downscale_4_5(G_buffer->data[i]);
        G_buffer->s.D++;
    }
}


void tt_dense_train(tensor_t *W,
                    const tensor_t *x,
                    const tensor_t *err_next,
                    tensor_t *err_prev,
                    tensor_t *G_buffer)
{
    size_t OUT = err_next->len;
    size_t IN  = x->len;

    /* make sure G_buffer has the right length */
    G_buffer->len = W->len;

    /* 1. gradient wrt weights: outer‑product ----------------------- */
    for (size_t r = 0; r < OUT; ++r) {
        for (size_t c = 0; c < IN; ++c) {
            int16_t g16 = (int16_t)err_next->data[r] * x->data[c];
            G_buffer->data[r*IN + c] = clip_int8(g16);
        }
    }
    G_buffer->s.S = err_next->s.S + x->s.S;
    G_buffer->s.U = err_next->s.U + x->s.U;
    G_buffer->s.D = err_next->s.D + x->s.D;

    /* 2. learning‑rate multiply (>>8) */
    for (size_t i = 0; i < G_buffer->len; ++i) {
        G_buffer->data[i] = clip_int8( shift_and_round32(G_buffer->data[i], LR_SHIFT) );
    }
    G_buffer->s.S -= LR_SHIFT;

    /* 3. align scales (Alg 3 lines 1–6) */
    align_scale(W, G_buffer);

    /* --- Alg 3 lines 8–11: margin bit‑width adjustment ---------- */
    {
        uint8_t b_g = 0, b_w = 0;
        /* compute bit‑width of gradient and weights */
        for (size_t i = 0; i < W->len; ++i) {
            int32_t gv = G_buffer->data[i];
            int32_t wv = W->data[i];
            uint8_t bg = eff_bitwidth32(gv);
            uint8_t bw = eff_bitwidth32(wv);
            if (bg > b_g) b_g = bg;
            if (bw > b_w) b_w = bw;
        }
        int8_t b = (int8_t)b_w - MARGIN;           /* target bit‑width */
        int8_t shift_adj = (int8_t)b_g - b;        /* how much to shift G */
        if (shift_adj > 0) {
            for (size_t i = 0; i < G_buffer->len; ++i) {
                G_buffer->data[i] = clip_int8(
                    shift_and_round32(G_buffer->data[i], shift_adj)
                );
            }
            G_buffer->s.S -= shift_adj;            /* update scale header */
        }
    }

    /* 4. SGD update ---------------------------- */
    for (size_t i = 0; i < W->len; ++i) {
        W->data[i] = clip_int8(W->data[i] - G_buffer->data[i]);
    }

    /* optional weight renorm ------------------ */
    int8_t maxw = 0;
    for (size_t i = 0; i < W->len; ++i)
        if (abs(W->data[i]) > maxw) maxw = abs(W->data[i]);

    if (maxw > 112) {
        for (size_t i = 0; i < W->len; ++i)
            W->data[i] = downscale_4_5(W->data[i]);
        W->s.D++;
    } else if (maxw < 32) {
        for (size_t i = 0; i < W->len; ++i)
            W->data[i] = upscale_4_3(W->data[i]);
        W->s.U++;
    }

    /* 5. error to previous layer: Wᵀ·err_next --------------------- */
    for (size_t c = 0; c < IN; ++c) {
        int32_t acc = 0;
        for (size_t r = 0; r < OUT; ++r) {
            acc += (int32_t)W->data[r*IN + c] * err_next->data[r];
        }
        err_prev->data[c] = clip_int8( shift_and_round32(acc, 7) );  /* quick */
    }
    err_prev->s.S = W->s.S + err_next->s.S - 7;
}
