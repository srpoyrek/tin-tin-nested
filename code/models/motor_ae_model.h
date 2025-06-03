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

#define MOTOR_AE_MODEL_PRNG_MIN (-63)
#define MOTOR_AE_MODEL_PRNG_MAX (-63)

#define LAYER1_ID (1)
#define LAYER2_ID (2)
#define LAYER3_ID (3)

/*----------------------------------------------------------------------*
 * Macro to declare a DenseLayer type specialized to IN×OUT dimensions.
 * It defines:
 *   - a packed buffer W_buf[OUT*IN]
 *   - a tensor_t   W (view into W_buf)
 *   - an activation buf a_buf[OUT]
 *   - a tensor_t   A (view into a_buf)
 *----------------------------------------------------------------------*/
#define DECLARE_DENSE_LAYER(NAME, IN, OUT)                    \
  enum { NAME##_WEIGHTS_SIZE = (IN) * (OUT),                  \
    NAME##_ACTIVATIONS_SIZE = (OUT)};                         \
  typedef struct {                                            \
    int8_t     W_buf[(IN)*(OUT)]; /* weight storage */        \
    tensor_t   W;                 /* tensor view of W_buf*/   \
    int8_t     A_buf[(OUT)];      /* activation storage */    \
    tensor_t   A;                 /* tensor view of A_buf*/   \
  } NAME##_t;

/* Use the macro to create three specific layer types: */
DECLARE_DENSE_LAYER(LAYER1, MOTOR_IN, MOTOR_H1)
DECLARE_DENSE_LAYER(LAYER2, MOTOR_H1, MOTOR_H2)
DECLARE_DENSE_LAYER(LAYER3, MOTOR_H2, MOTOR_OUT)

#define DECLARE_INOUT_LAYER(NAME, SIZE)                       \
  enum { NAME##_SIZE = SIZE };                              \
  typedef struct {                                            \
    int8_t     INOUT_buf[SIZE];   /* weight storage */        \
    tensor_t   INOUT;             /* tensor view of W_buf*/   \
  } NAME##_t;

DECLARE_INOUT_LAYER(INPUT,  MOTOR_IN)
DECLARE_INOUT_LAYER(OUTPUT, MOTOR_OUT)


typedef struct {
  const TensorBackend_t *ops;   /* TT vs nested‑TT vs 4‑bit etc. */
  
  /* I/O buffers and views */
  INPUT_t   input;
  OUTPUT_t  output;  
  /* Hidden layers */
  LAYER1_t  layer1;
  LAYER2_t  layer2;
  LAYER3_t  layer3;

  int32_t rng;
} tt_motor_ae_model_t;
  

void motor_ae_model_init(void * model,const TensorBackend_t *backend, uint32_t seed);
uint32_t motor_ae_model_forward(void * model, const tensor_t *x);
void motor_ae_model_backward(void * model, const tensor_t *x);

#endif // TT_MOTOR_AE_MODEL_H
