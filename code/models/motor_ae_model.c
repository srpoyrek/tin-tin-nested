#include "motor_ae_model.h"
#include "tt_math.h"
#include "tt_types.h"
#include <stdlib.h>

static int8_t rng_int8(uint32_t *s) {
    *s = *s * 1664525u + 1013904223u;
    return (int8_t)(((*s >> 24) & 0x7F) - 63);
}

void tt_motor_ae_model_init(tt_motor_ae_model_t *m,
                            const TensorBackend_t *backend,
                            uint32_t seed)
{
    m->ops = backend;
    uint32_t s = seed ? seed : 1;
    // randomize weights
    for (size_t i = 0; i < sizeof(m->W1_buf); ++i) m->W1_buf[i] = rng_int8(&s);
    for (size_t i = 0; i < sizeof(m->W2_buf); ++i) m->W2_buf[i] = rng_int8(&s);
    for (size_t i = 0; i < sizeof(m->W3_buf); ++i) m->W3_buf[i] = rng_int8(&s);
    // bind tensors
    tt_tensor_init(&m->W1, m->W1_buf, sizeof(m->W1_buf));
    tt_tensor_init(&m->W2, m->W2_buf, sizeof(m->W2_buf));
    tt_tensor_init(&m->W3, m->W3_buf, sizeof(m->W3_buf));
    tt_tensor_init(&m->h1, m->h1_buf, 24);
    tt_tensor_init(&m->h2, m->h2_buf, 24);
    tt_tensor_init(&m->y,  m->y_buf,  32);
}

uint32_t tt_motor_ae_model_forward(tt_motor_ae_model_t *m,
                                   const tensor_t *x)
{
    // encoder
    m->ops->dense_forward(&m->W1, x, &m->h1);
    m->ops->dense_forward(&m->W2, &m->h1, &m->h2);
    // decoder
    m->ops->dense_forward(&m->W3, &m->h2, &m->y);
    // compute SSE
    uint32_t sse = 0;
    for (size_t i = 0; i < 32; ++i) {
        int16_t d = (int16_t)x->data[i] - m->y_buf[i];
        sse += (uint32_t)(d * d);
    }
    return sse;
}

void tt_motor_ae_model_backward(tt_motor_ae_model_t *m,
                                const tensor_t *x)
{
    int8_t err3_buf[32]; tensor_t err3;
    tt_tensor_init(&err3, err3_buf, 32);
    for (size_t i = 0; i < 32; ++i)
        err3_buf[i] = clip_int8((int16_t)x->data[i] - m->y_buf[i]);
    
    int8_t err2_buf[24]; tensor_t err2;
    tt_tensor_init(&err2, err2_buf, 24);
    int8_t G3_buf[32*24]; tensor_t G3;
    tt_tensor_init(&G3, G3_buf, 32*24);
    m->ops->dense_train(&m->W3, &m->h2, &err3, &err2, &G3);
    
    int8_t err1_buf[24]; tensor_t err1;
    tt_tensor_init(&err1, err1_buf, 24);
    int8_t G2_buf[24*24]; tensor_t G2;
    tt_tensor_init(&G2, G2_buf, 24*24);
    m->ops->dense_train(&m->W2, &m->h1, &err2, &err1, &G2);
    
    int8_t dummy_buf[24]; tensor_t dummy;
    tt_tensor_init(&dummy, dummy_buf, 24);
    int8_t G1_buf[24*32]; tensor_t G1;
    tt_tensor_init(&G1, G1_buf, 24*32);
    m->ops->dense_train(&m->W1, x, &err1, &dummy, &G1);
    
    // clear temps
    tt_tensor_clear(&err3);
    tt_tensor_clear(&err2);
    tt_tensor_clear(&err1);
    tt_tensor_clear(&dummy);
    tt_tensor_clear(&G3);
    tt_tensor_clear(&G2);
    tt_tensor_clear(&G1);
}

void tt_motor_ae_model_destroy(tt_motor_ae_model_t *m)
{
    tt_tensor_clear(&m->W1);
    tt_tensor_clear(&m->W2);
    tt_tensor_clear(&m->W3);
    tt_tensor_clear(&m->h1);
    tt_tensor_clear(&m->h2);
    tt_tensor_clear(&m->y);
}
