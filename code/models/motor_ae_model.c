#include "motor_ae_model.h"
#include "tt_math.h"
#include "tt_types.h"
#include "prng.h"
#include <stdlib.h>


void motor_ae_model_init(tt_motor_ae_model_t *m, const TensorBackend_t *backend, uint32_t seed) {
    // return if null 
    if(!m || backend) return;
    // assign the backend tensor operations
    m->ops = backend;
    // init the random number generation
    prng_init(&m->rng, seed);

    // init the input & output tensors
    tt_tensor_init(&m->input,  m->in_buf,  MOTOR_IN);
    tt_tensor_init(&m->output, m->out_buf, MOTOR_OUT);

    // init the layer1 tensors
    size_t l1_size = MOTOR_IN * MOTOR_H1;
    tt_tensor_init(&m->layer1.W, m->layer1.W_buf, l1_size);
    tt_tensor_init(&m->layer1.A, m->layer1.A_buf, MOTOR_H1);
    // random allocation of weights
    for (size_t i = 0; i < l1_size; ++i) {
        m->layer1.W_buf[i] = prng_rand_range(&m->rng, -63, +63);        
    }

    // init the layer2 tensors
    size_t l2_size = MOTOR_IN * MOTOR_H1;
    tt_tensor_init(&m->layer2.W, m->layer2.W_buf, l2_size);
    tt_tensor_init(&m->layer2.A, m->layer2.A_buf, MOTOR_H2);
    // random allocation of weights
    for (size_t i = 0; i < l1_size; ++i) {
        m->layer2.W_buf[i] = prng_rand_range(&m->rng, -63, +63);        
    }

    // init the layer3 tensors
    size_t l3_size = MOTOR_IN * MOTOR_H1;
    tt_tensor_init(&m->layer3.W, m->layer3.W_buf, l3_size);
    tt_tensor_init(&m->layer3.A, m->layer3.A_buf, MOTOR_OUT);
    // random allocation of weights
    for (size_t i = 0; i < l3_size; ++i) {
        m->layer2.W_buf[i] = prng_rand_range(&m->rng, -63, +63);        
    }

    return;
}


/*----------------------------------------------------------------------*
 * Forward pass just loops over each layer, using its own buffers.
 *----------------------------------------------------------------------*/
uint32_t tt_motor_ae_forward(tt_motor_ae_model_t *m, const int8_t *in_data) {
    /* copy input */
    memcpy(m->in_buf, in_data, MOTOR_IN);

    /* scratch for dot-product accumulations */
    int32_t acc_buf[MOTOR_OUT];

    /* Layer 1 */
    m->ops->dense_forward(&m->layer1.W, &m->input, &m->layer1.A, acc_buf);
    /* Layer 2 */
    m->ops->dense_forward(&m->layer2.W, &m->layer1.A, &m->layer2.A, acc_buf);
    /* Layer 3 (reconstruction) */
    m->ops->dense_forward(&m->layer3.W, &m->layer2.A, &m->layer3.A, acc_buf);

    /* copy to output buffer */
    memcpy(m->out_buf, m->layer3.A.data, MOTOR_OUT);

    /* compute SSE */
    uint32_t sse = 0;
    for (size_t i = 0; i < MOTOR_OUT; ++i) {
        int16_t d = m->in_buf[i] - m->out_buf[i];
        sse += (uint32_t)(d*d);
    }
    return sse;
}

void tt_motor_ae_backward(tt_motor_ae_model_t *m)
{
    /*--- prepare error and gradient buffers per layer ---*/
    int8_t err3_buf[MOTOR_OUT]; tensor_t err3;
    int8_t err2_buf[MOTOR_H2]; tensor_t err2;
    int8_t err1_buf[MOTOR_H1]; tensor_t err1;

    int8_t G3_buf[MOTOR_OUT * MOTOR_H2]; tensor_t G3;
    int8_t G2_buf[MOTOR_H2 * MOTOR_H1]; tensor_t G2;
    int8_t G1_buf[MOTOR_H1 * MOTOR_IN]; tensor_t G1;

    /* bind error and gradient tensors */
    tt_tensor_init(&err3, err3_buf, MOTOR_OUT);
    tt_tensor_init(&err2, err2_buf, MOTOR_H2);
    tt_tensor_init(&err1, err1_buf, MOTOR_H1);
    tt_tensor_init(&G3,   G3_buf,   MOTOR_OUT * MOTOR_H2);
    tt_tensor_init(&G2,   G2_buf,   MOTOR_H2 * MOTOR_H1);
    tt_tensor_init(&G1,   G1_buf,   MOTOR_H1 * MOTOR_IN);

    /* 1) output-layer error: err3 = input - reconstruction */
    for (size_t i = 0; i < MOTOR_OUT; ++i) {
        err3_buf[i] = clip_int8((int16_t)m->in_buf[i] - m->layer3.A.data[i]);
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

    /* clear temporary tensors */
    tt_tensor_clear(&err3);
    tt_tensor_clear(&err2);
    tt_tensor_clear(&err1);
    tt_tensor_clear(&G3);
    tt_tensor_clear(&G2);
    tt_tensor_clear(&G1);
    tt_tensor_clear(&dummy);
    return;
}
