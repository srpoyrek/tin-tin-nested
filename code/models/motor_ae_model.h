/*============================================================
 * File: tt_motor_ae_model.h
 * Generic motor AE model using pluggable backend
 *============================================================*/
#ifndef TT_MOTOR_AE_MODEL_H
#define TT_MOTOR_AE_MODEL_H

#include "tt_tensor_backend.h"
#include "tt_types.h"
#include <stdint.h>

#define MOTOR_IN (32)
#define MOTOR_H1 (24)
#define MOTOR_H2 (24)
#define MOTOR_OUT (MOTOR_IN)

/*----------------------------------------------------------------------*
 * Macro to declare a DenseLayer type specialized to IN×OUT dimensions.
 * It defines:
 *   - a packed buffer W_buf[OUT*IN]
 *   - a tensor_t   W (view into W_buf)
 *   - an activation buf a_buf[OUT]
 *   - a tensor_t   A (view into a_buf)
 *----------------------------------------------------------------------*/
#define DECLARE_DENSE_LAYER(NAME, IN, OUT)                     \
  typedef struct {                                            \
    int8_t     W_buf[(IN)*(OUT)];    /* weight storage */     \
    tensor_t   W;                     /* tensor view of W_buf*/\
    int8_t     A_buf[(OUT)];         /* activation storage */ \
    tensor_t   A;                     /* tensor view of A_buf*/\
  } NAME;

/* Use the macro to create three specific layer types: */
DECLARE_DENSE_LAYER(Layer1_t, MOTOR_IN, MOTOR_H1)
DECLARE_DENSE_LAYER(Layer2_t, MOTOR_H1, MOTOR_H2)
DECLARE_DENSE_LAYER(Layer3_t, MOTOR_H2, MOTOR_OUT)


typedef struct {
    const TensorBackend_t *ops;   /* TT vs nested‑TT vs 4‑bit etc. */
  
    /* I/O buffers and views */
    int8_t    in_buf [MOTOR_IN];
    tensor_t  input;
    int8_t    out_buf[MOTOR_OUT];
    tensor_t  output;
  
    /* Hidden layers */
    Layer1_t  layer1;
    Layer2_t  layer2;
    Layer3_t  layer3;

    int32_t rng;
  } tt_motor_ae_model_t;
  

void motor_ae_model_init(tt_motor_ae_model_t *m,const TensorBackend_t *backend, uint32_t seed);
uint32_t motor_ae_model_forward(tt_motor_ae_model_t *m, const tensor_t *x);
void motor_ae_model_backward(tt_motor_ae_model_t *m, const tensor_t *x);

#endif // TT_MOTOR_AE_MODEL_H
