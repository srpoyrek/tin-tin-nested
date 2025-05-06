/*============================================================
 * File: tt_motor_ae_model.h
 * Generic motor AE model using pluggable backend
 *============================================================*/
#ifndef TT_MOTOR_AE_MODEL_H
#define TT_MOTOR_AE_MODEL_H

#include "tt_tensor_backend.h"
#include "tt_types.h"
#include <stdint.h>


typedef struct {
    const TensorBackend_t *ops;  // chosen backend
    // static buffers:
    int8_t W1_buf[24 * 32];
    int8_t W2_buf[24 * 24];
    int8_t W3_buf[32 * 24];
    int8_t h1_buf[24];
    int8_t h2_buf[24];
    int8_t y_buf[32];
    // tensor views:
    tensor_t W1, W2, W3;
    tensor_t h1, h2, y;
} tt_motor_ae_model_t;

void tt_motor_ae_model_init(tt_motor_ae_model_t *m,
                            const TensorBackend_t *backend,
                            uint32_t seed);
uint32_t tt_motor_ae_model_forward(tt_motor_ae_model_t *m,
                                   const tensor_t *x);
void tt_motor_ae_model_backward(tt_motor_ae_model_t *m,
                                const tensor_t *x);
void tt_motor_ae_model_destroy(tt_motor_ae_model_t *m);

#endif // TT_MOTOR_AE_MODEL_H
