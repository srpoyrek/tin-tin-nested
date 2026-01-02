/**
 * @file tt_types.h
 * @brief Type definitions for Tin-Tin tensors
 * @author Shreyas Poyrekar
 * @date May 7, 2025
 */

#ifndef TT_TYPES_H
#define TT_TYPES_H

#include "scale.h"
#include <stddef.h>
#include <stdint.h>

/**
 * @brief Tensor shape descriptor
 *
 * For flexibility, supports up to 4 dimensions:
 * - 1D: [len] (vectors)
 * - 2D: [rows, cols] (matrices, batches of vectors)
 * - 3D: [batch, rows, cols] (batched matrices)
 * - 4D: [batch, channels, height, width] (images)
 */
typedef struct {
    size_t ndim;       /**< Number of dimensions (1-4) */
    size_t shape[4];   /**< Dimensions: [d0, d1, d2, d3] */
} tensor_shape_t;

/**
 * @brief Generic N-D tensor with shape
 *
 * Data interpretation:
 * - Inputs/Activations: uint8_t (non-negative after ReLU)
 * - Weights/Gradients: int8_t (can be negative)
 * - Accumulator buffers: int32_t (intermediate results)
 *
 * Storage is always **row-major** (C-style) flattened.
 */
typedef struct {
    void *data;           /**< Payload (int8_t*, uint8_t*, or int32_t*) */
    tensor_shape_t shape; /**< Shape descriptor */
    size_t len;           /**< Total number of elements (product of shape) */
    scale_t s;            /**< Scale exponents */
} tensor_t;

/* ==================== Tensor ==================== */

/**
 * @brief Initialize tensor with data and length
 */
void tt_tensor_init(const tensor_t *t, tensor_shape_t shape, void *data, size_t len);

/**
 * @brief Clear tensor (set to NULL/0)
 */
void tt_tensor_clear(tensor_t *t);

/**
 * @brief Get tensor shape
 */
tensor_shape_t tt_tensor_get_shape(const tensor_t * const t);

/* ==================== Debug Printing ==================== */

#ifdef TT_BIG_MACHINE_DEBUG_ENABLE
/**
 * @brief Print tensor contents (first 8 elements)
 */
void tt_tensor_print(const char *name, const tensor_t *t);
#endif

#endif /* TT_TYPES_H */
