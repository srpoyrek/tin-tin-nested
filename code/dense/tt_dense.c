/**
 * @file tt_dense.c
 * @brief Tin-Tin dense layer implementation with quantized training
 * @details This file implements forward and backward passes for dense layers
 *          using 8-bit integer arithmetic with dynamic scaling as described
 *          in the Tin-Tin paper.
 * @author Shreyas Poyrekar
 * @date 05-29-2025
 * @license MIT
 */

#include "tt_dense.h"
#include "tt_math.h"
#include "matrix.h"
#include "activations.h"
#include <stdlib.h>
#include <string.h>

/* ==================== Constants ==================== */

/** @brief Learning rate as right shift: LR = 1/256 = 2^(-8) */
#define LR_SHIFT 8    

/** @brief Bit-width safety margin for gradient scaling (Algorithm 3, line 10) */
#define MARGIN   2    

/** @brief Lower threshold for dynamic range adjustment: 2^(CHAR_BIT-1)/4 = 32 */
#define T_LOW   (2 ^ (CHAR_BIT - 1) / 4)

/** @brief Upper threshold for dynamic range adjustment: 2^(CHAR_BIT-1)*7/8 = 112 */
#define T_HIGH  ((2 ^ (CHAR_BIT - 1) * 7) / 8)

/** @brief 7 bits for int8_t magnitude */
#define TARGET_BITS (7)

/* ==================== Static Function Declarations ==================== */

/**
 * @brief Align gradient and weight scales (Algorithm 3, lines 1-6)
 * @param W Weight tensor (modified)
 * @param G_buffer Gradient buffer (modified)
 */
static void align_scale(tensor_t *W, tensor_t *G_buffer);

/**
 * @brief Perform dynamic range adjustment on tensor
 * @param T Tensor to adjust
 */
static void s_dynamic_scale_adjustment(tensor_t *T);

/**
 * @brief Find maximum absolute value in tensor
 * @param Y Input tensor
 * @return Maximum absolute value as int8_t
 */
static inline int8_t s_find_max_abs_value(const tensor_t *Y);

/**
 * @brief Apply shift, round, and clip operations to tensor
 * @param Y Target tensor (modified in-place)
 * @param sfunc Shift-round function
 * @param ksh Shift amount
 * @param cfunc Clip function
 */
static inline void s_apply_shift_round_clip(tensor_t *Y, shift_round_t sfunc, uint8_t ksh, clip_t cfunc);

/**
 * @brief Apply activation function to int32 buffer
 * @param buffer Target buffer (modified in-place)
 * @param len Number of elements
 * @param func Activation function
 */
static inline void s_apply_activation(int32_t *buffer, size_t len, Activation_i8_t func);

/**
 * @brief Apply scaling function to tensor
 * @param Y Target tensor (modified in-place)
 * @param func Scaling function
 */
static inline void s_apply_scale_by(tensor_t *Y, scale_by_t func);

/**
 * @brief Forward pass for dense layer: Y = ReLU(W * X)
 * @details Implements Algorithm 1 from Tin-Tin paper with:
 *          1. Matrix multiplication in int32
 *          2. ReLU activation
 *          3. Bit-width analysis and shifting
 *          4. Scale combination and adjustment
 *          5. Dynamic range optimization
 * 
 * @param W Weight tensor (rows=output_size, cols=input_size, flattened)
 * @param X Input activation tensor
 * @param Y Output activation tensor (result)
 * @param acc_buffer Accumulation buffer for intermediate int32 results
 * @param acc_size Size of accumulation buffer (must equal Y->len)
 * 
 * @pre W, X, Y, acc_buffer must not be NULL
 * @pre acc_size must equal Y->len
 * @pre W->len must equal (Y->len * X->len)
 * 
 * @post Y contains quantized activations with updated scale
 */
void tt_dense_forward(const tensor_t *W, const tensor_t *X, tensor_t *Y, int32_t * acc_buffer, size_t acc_size) {

    // Input validation
    if(!W || !X || !Y || !acc_buffer || acc_size != Y->len) return;
    
    // raw int32 matrix-vector multiplication
    matrix_mul(W, X, acc_buffer);

    // apply ReLU to the int32 accumulators
    s_apply_activation(acc_buffer, Y->len, relu_i8);

    // get the bits to shift by effective max bit width
    uint8_t max_bitwidth = eff_bitwidth_array(acc_buffer,Y->len);
    uint8_t ksh = (max_bitwidth > TARGET_BITS) ? (max_bitwidth - TARGET_BITS) : 0;

    // shift and round by 32 and clip by int8_t
    s_apply_shift_round_clip(Y, shift_and_round32, ksh, clip_int8);

    // combine the scales of weights and Activations 
    scale_combine(&Y->s, &W->s, &X->s);

    // shift the scale of Y
    scale_shift(&Y->s, -(int8_t)ksh);

    // dynamic range scale adjustment 
    s_dynamic_scale_adjustement(Y);

#ifdef TENSOR_USE_NESTED
    // roll up scale
    scale_rollup(&Y->s);
#endif
    return;
}

/**
 * @brief Training pass for dense layer with backward propagation
 * @details Implements Algorithm 2 and 3 from Tin-Tin paper:
 *          1. Gradient computation (outer product)
 *          2. Learning rate application
 *          3. Scale alignment
 *          4. Bit-width margin adjustment
 *          5. SGD weight update
 *          6. Weight renormalization
 *          7. Error backpropagation
 * 
 * @param W Weight tensor (modified in-place)
 * @param X Input activation tensor
 * @param err_next Error signal from next layer
 * @param err_prev Error signal to previous layer (output)
 * @param G_buffer Gradient buffer (temporary storage)
 * 
 * @pre All pointers must not be NULL
 * @pre G_buffer->len must equal W->len
 * @pre W dimensions: (err_next->len × X->len)
 * 
 * @post W contains updated weights
 * @post err_prev contains error for previous layer
 */
void tt_dense_train(tensor_t *W, const tensor_t *X, const tensor_t *err_next, tensor_t *err_prev, tensor_t *G_buffer) {
    
    // Input validation
    if(!W || !X || !err_next || !err_prev || !G_buffer) return;
    if(G_buffer->len != W->len) return;

    size_t OUT = err_next->len;
    size_t IN  = X->len;

    // get the gradient by multiplying the input with the error from upper layer
    for (size_t r = 0; r < OUT; ++r) {
        for (size_t c = 0; c < IN; ++c) {
            int16_t g16 = (int16_t)err_next->data[r] * X->data[c];
            G_buffer->data[r * IN + c] = clip_int8(g16);
        }
    }

    // Apply learning rate (divide by 256)
    s_apply_shift_round_clip(G_buffer, shift_and_round32, LR_SHIFT, clip_int8);
    // combine the scales
    scale_combine(&G_buffer->s, &err_next->s, &X->s);
    // shift the scale
    scale_shift(&G_buffer->s, -LR_SHIFT);

    // Align scales between weights and gradients 
    align_scale(W, G_buffer);

    // Bit-width margin adjustment
    uint8_t b_w = eff_bitwidth_array(W->data, W->len);
    uint8_t b_g = eff_bitwidth_array(G_buffer->data, G_buffer->len);

    int8_t target_bitwidth = (int8_t)b_w - MARGIN;
    int8_t shift_adj = (int8_t)b_g - target_bitwidth;

     if (shift_adj > 0) {
        s_apply_shift_round_clip(G_buffer, shift_and_round32, shift_adj, clip_int8);
        scale_shift(&G_buffer->s, -shift_adj);
    }

    // SGD weight update (W = W - G)
    for (size_t i = 0; i < W->len; ++i) {
        int16_t new_weight = (int16_t)W->data[i] - G_buffer->data[i];
        W->data[i] = clip_int8(new_weight);
    }


    // Weight renormalization for numerical stability
    s_dynamic_scale_adjustement(W);

    // Compute error for previous layer (W^T * err_next)
    for (size_t c = 0; c < IN; ++c) {
        int32_t acc = 0;
        for (size_t r = 0; r < OUT; ++r) {
            acc += (int32_t)W->data[r * IN + c] * err_next->data[r];
        }
        err_prev->data[c] = clip_int8(shift_and_round32(acc, 7));
    }

    // Update error scale for previous layer
    scale_combine(&err_prev->s, &W->s, &err_next->s);
    scale_shift(&err_prev->s, -7);

    return;
}

/* ==================== Static Function Implementations ==================== */

/**
 * @brief Perform dynamic range adjustment based on maximum absolute value
 * @param T Tensor to adjust (modified in-place)
 */
static void s_dynamic_scale_adjustment(tensor_t *T) {
    if (!T) return;
    
    int8_t max_val = s_find_max_abs_value(T);
    
    if (max_val > T_HIGH) {
        // Values too large - scale down by factor ~0.8
        s_apply_scale_by(T, downscale_4_5);
        scale_down(&T->s);   
    } else if (max_val < T_LOW) {
        // Values too small - scale up by factor ~1.33
        s_apply_scale_by(T, upscale_4_3);
        scale_up(&T->s);   
    }
}

/**
 * @brief Align gradient and weight scales (Algorithm 3, lines 1-6)
 * @param W Weight tensor
 * @param G_buffer Gradient buffer (modified to match W's scale)
 */
static void align_scale(tensor_t *W, tensor_t *G_buffer) {
    if (!W || !G_buffer) return;
    
    // Calculate scale differences
    int8_t dS = W->s.S - G_buffer->s.S;
    int8_t dU = W->s.U - G_buffer->s.U;
    int8_t dD = W->s.D - G_buffer->s.D;

    // Align power-of-2 shifts
    if (dS > 0) {
        // Left shift gradients
        for (size_t i = 0; i < G_buffer->len; ++i) {
            int16_t shifted = (int16_t)G_buffer->data[i] << dS;
            G_buffer->data[i] = clip_int8(shifted);
        }
        G_buffer->s.S += dS;
    } else if (dS < 0) {
        // Right shift gradients with rounding
        dS = -dS;
        s_apply_shift_round_clip(G_buffer, shift_and_round32, 
                                dS, clip_int8);
        G_buffer->s.S -= dS;
    }

    // Align up-scale counts
    while (dU-- > 0) {
        s_apply_scale_by(G_buffer, upscale_4_3);
        G_buffer->s.U++;
    }
    
    // Align down-scale counts
    while (dD-- > 0) {
        s_apply_scale_by(G_buffer, downscale_4_5);
        G_buffer->s.D++;
    }
}


/**
 * @brief Apply activation function element-wise to int32 buffer
 * @param buffer Target buffer (modified in-place)
 * @param len Number of elements
 * @param func Activation function pointer
 */
static inline void s_apply_activation(int32_t *buffer, size_t len, 
                                     Activation_i8_t func) {
    if (!buffer || !func) return;
    
    for (size_t i = 0; i < len; ++i) {
        buffer[i] = (int32_t)func((int8_t)buffer[i]);
    }
}

/**
 * @brief Find maximum absolute value in tensor
 * @param Y Input tensor
 * @return Maximum absolute value, or 0 if Y is NULL
 */
static inline int8_t s_find_max_abs_value(const tensor_t *Y) {
    if (!Y || !Y->data) return 0;
    
    int8_t max_val = 0;
    for (size_t i = 0; i < Y->len; ++i) {
        int8_t abs_val = abs(Y->data[i]);
        max_val = max(max_val, abs_val);
    }
    return max_val;
}

/**
 * @brief Apply shift, round, and clip operations to tensor
 * @param Y Target tensor (modified in-place)
 * @param sfunc Shift-round function
 * @param ksh Number of bits to shift
 * @param cfunc Clipping function
 */
static inline void s_apply_shift_round_clip(tensor_t *Y, shift_round_t sfunc,
                                           uint8_t ksh, clip_t cfunc) {
    if (!Y || !Y->data || !sfunc || !cfunc) return;
    
    for (size_t i = 0; i < Y->len; ++i) {
        int32_t shifted = sfunc((int32_t)Y->data[i], ksh);
        Y->data[i] = cfunc(shifted);
    }
}

/**
 * @brief Apply scaling function element-wise to tensor
 * @param Y Target tensor (modified in-place)
 * @param len Number of elements to process
 * @param func Scaling function pointer
 */
static inline void s_apply_scale_by(tensor_t *Y, scale_by_t func) {
    if (!Y || !Y->data || !func) return;
    
    for (size_t i = 0; i < Y->len; ++i) {
        Y->data[i] = func(Y->data[i]);
    }
}