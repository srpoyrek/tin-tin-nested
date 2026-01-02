/**
 * @file tt_dense.h
 * @brief Dense layer forward and training passes
 * @author Shreyas Poyrekar
 * @date May 7, 2025
 */

#ifndef TT_DENSE_H
#define TT_DENSE_H

#include "tt_types.h"
#include "matrix.h"

/**
 * @brief Single-element activation function
 */
typedef int8_t (*activation_func_t)(int8_t x);

/**
 * @brief Apply activation to int32 buffer
 */
typedef void (*apply_activation_t)(int32_t *buffer, size_t len, activation_func_t func);

/**
 * @brief Configuration for dense layer operations
 */
typedef struct {
    matrix_mul_func matrix_mul_func;          /**< Matrix multiply callback */
    activation_func_t fd_activation_func;     /**< Element-wise activation (forward) */
    apply_activation_t apply_activation_func; /**< Batch activation applier */
} tensor_cfg_t;

/**
 * @brief Forward pass context
 */
typedef struct
{
    const tensor_t *W;       /**< Weights (OUT × IN) */
    const tensor_t *X;       /**< Input (IN) */
    tensor_t *Y;             /**< Output (OUT) */
    void *acc_buffer;        /**< Accumulator int32_t[OUT] */
} tensor_forward_t;

/**
 * @brief Training pass context
 */
typedef struct
{
    tensor_t *W;                /**< Weights (OUT × IN) - modified in-place */
    const tensor_t *X;          /**< Input activations (IN) */
    const tensor_t *err_next;   /**< Error from next layer (OUT) */
    tensor_t *err_prev;         /**< Error to previous layer (IN) - output */
    tensor_t *G_buffer;         /**< Gradient buffer (OUT × IN) - temp storage */
} tensor_train_t;

/**
 * @brief Forward pass: Y = activation(W × X)
 *
 * @param ctx Forward context with all tensors and buffers
 * @param cfg Configuration with function callbacks
 */
void tt_dense_forward(const tensor_forward_t *ctx, const tensor_cfg_t *cfg);

/**
 * @brief Training pass: compute gradients and backprop error
 *
 * @param ctx Training context with all tensors
 * @param cfg Configuration with function callbacks
 *
 * @post ctx->W is updated with gradient descent
 * @post ctx->err_prev contains backpropagated error
 */
void tt_dense_train(const tensor_train_t *ctx, const tensor_cfg_t *cfg);

#endif /* TT_DENSE_H */
