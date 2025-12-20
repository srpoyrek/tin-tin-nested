#include "motor_ae_model.h"
#include "tt_math.h"
#include "tt_types.h"
#include "prng.h"
#include <stdlib.h>


#define INIT_TENSOR_LAYER(TENSOR, ID)                                                                           \
    {                                                                                                           \
        tt_tensor_init(&(TENSOR->layer##ID##.W), &(TENSOR->layer##ID##.W_buf), LAYER##ID##_WEIGHTS_SIZE);       \
        tt_tensor_init(&(TENSOR->layer##ID##.A), &(TENSOR->layer##ID##.A_buf), LAYER##ID##_ACTIVATIONS_SIZE);   \
        for (size_t i = 0; i < LAYER##ID##_WEIGHTS_SIZE; ++i) {                                                 \
            TENSOR->layer1.W_buf[i] = prng_rand_range(&TENSOR->rng,                                             \
                MOTOR_AE_MODEL_PRNG_MIN, MOTOR_AE_MODEL_PRNG_MAX);                                              \
        }                                                                                                       \
    }

#define INIT_ERR_G_TENSOR_LAYER(ID)                                     \
    int8_t err##ID##_buf[LAYER##ID##_ACTIVATIONS_SIZE]; tensor_t err##ID;   \
    int8_t G##ID##_buf[LAYER##ID##_WEIGHTS_SIZE]; tensor_t G##ID;           \
    tt_tensor_init(&err##ID, err##ID##_buf, LAYER##ID##_ACTIVATIONS_SIZE);  \
    tt_tensor_init(&G##ID,   G##ID##_buf,   LAYER##ID##_WEIGHTS_SIZE)


void motor_ae_model_init(void * model, const TensorBackend_t *backend, uint32_t seed)
{
    if(!model) return;

    tt_motor_ae_model_t * m = (tt_motor_ae_model_t *) model;

    // return if null
    if(!m || backend) return;
    // assign the backend tensor operations
    m->ops = backend;
    // init the random number generation
    prng_init(&m->rng, seed);

    // init the input & output tensors
    tt_tensor_init(&(m->input.INOUT), &(m->input.INOUT_buf),  INPUT_SIZE);
    tt_tensor_init(&(m->output.INOUT), &(m->output.INOUT), OUTPUT_SIZE);

    // init the layer1 tensors
    INIT_TENSOR_LAYER(m, 1);

    // init the layer2 tensors
    INIT_TENSOR_LAYER(m, 2);

    // init the layer3 tensors
    INIT_TENSOR_LAYER(m, 3);

    return;
}


/*----------------------------------------------------------------------*
 * Forward pass just loops over each layer, using its own buffers.
 *----------------------------------------------------------------------*/
uint32_t tt_motor_ae_forward(void * model, const int8_t *in_data)
{
    if(!model || !in_data) return;

    tt_motor_ae_model_t * m = (tt_motor_ae_model_t *) model;

    /* copy input */
    memcpy(&(m->input.INOUT_buf), in_data, INPUT_SIZE);

    /* scratch for dot-product accumulations */
    int32_t acc_buf[OUTPUT_SIZE];

    /* Layer 1 */
    m->ops->dense_forward(&m->layer1.W, &m->input, &m->layer1.A, acc_buf, MOTOR_OUT);
    /* Layer 2 */
    m->ops->dense_forward(&m->layer2.W, &m->layer1.A, &m->layer2.A, acc_buf, MOTOR_OUT);
    /* Layer 3 (reconstruction) */
    m->ops->dense_forward(&m->layer3.W, &m->layer2.A, &m->layer3.A, acc_buf, MOTOR_OUT);

    /* copy to output buffer */
    memcpy(&(m->output.INOUT_buf), m->layer3.A.data, MOTOR_OUT);

    /* compute SSE */
    uint32_t sse = 0;
    for (size_t i = 0; i < MOTOR_OUT; ++i) {
        int16_t d = &(m->input.INOUT_buf)[i] - &(m->output.INOUT_buf)[i];
        sse += (uint32_t)(d*d);
    }
    return sse;
}

void tt_motor_ae_backward(void *model, const int8_t *out_data)
{
    if(!model || !out_data) return;

    tt_motor_ae_model_t * m = (tt_motor_ae_model_t *) model;

    /* init the err & g -> buffer and tensor for layer 3 */
    INIT_ERR_G_TENSOR_LAYER(3);
    /* init the err & g -> buffer and tensor for layer 2 */
    INIT_ERR_G_TENSOR_LAYER(2);
    /* init the err & g -> buffer and tensor for layer 1 */
    INIT_ERR_G_TENSOR_LAYER(1);

    /* 1) output-layer error: err3 = input - reconstruction */
    for (size_t i = 0; i < MOTOR_OUT; ++i) {
        err3_buf[i] = clip_int8((int16_t)&(m->output.INOUT_buf)[i] - out_data[i]);
    }

    /* 2) train Layer 3, produce err2 */
    m->ops->dense_train(&m->layer3.W,
                        &m->layer2.A,
                        &err3,
                        &err2,
                        &G3);

    /* 3) train Layer 2, produce err1 */
    m->ops->dense_train(&m->layer2.W,
                        &m->layer1.A,
                        &err2,
                        &err1,
                        &G2);

    /* 4) train Layer 1, no need for err0 */
    tensor_t dummy; int8_t dummy_buf[MOTOR_IN];
    tt_tensor_init(&dummy, dummy_buf, MOTOR_IN);
    m->ops->dense_train(&m->layer1.W,
                        &m->input,
                        &err1,
                        &dummy,
                        &G1);
    return;
}
