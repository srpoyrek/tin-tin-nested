/*============================================================
 * File: tt_motor_ae_model.h
 * API for motor bearing fault auto-encoder (static buffers)
 * Input: window of 32 ints
 * Architecture: 32 -> 24 -> 24 -> 32
 *============================================================*/
#ifndef TT_MOTOR_AE_MODEL_H
#define TT_MOTOR_AE_MODEL_H

#include "tt_types.h"
#include <stdint.h>

/* model struct with static data buffers */
typedef struct {
    /* weight buffers */
    int8_t W1_buf[24 * 32];
    int8_t W2_buf[24 * 24];
    int8_t W3_buf[32 * 24];
    /* activation buffers */
    int8_t h1_buf[24];
    int8_t h2_buf[24];
    int8_t y_buf[32];
    /* tensor views */
    tensor_t W1, W2, W3;
    tensor_t h1, h2, y;
} tt_motor_ae_model_t;

/**
 * Initialize model: bind buffers via tt_tensor_init and randomize weights
 */
void tt_motor_ae_model_init(tt_motor_ae_model_t *m, uint32_t seed);

/**
 * Forward pass: reconstruct input x (length 32), return SSE
 */
uint32_t tt_motor_ae_model_forward(tt_motor_ae_model_t *m,
                                   const tensor_t *x);

/**
 * Backward pass + SGD update (lr = 1/256)
 */
void tt_motor_ae_model_backward(tt_motor_ae_model_t *m,
                                const tensor_t *x);

/**
 * Tear down model: clear tensor views
 */
void tt_motor_ae_model_destroy(tt_motor_ae_model_t *m);

#endif // TT_MOTOR_AE_MODEL_H