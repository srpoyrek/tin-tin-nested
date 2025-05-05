#include <stdlib.h>
#include <string.h>
#include "tin_ae.h"
#include "tin_dense.h"
#include "tin_train.h"
#include "tin_math.h"

/* ------------ helpers -------------------------------------------------- */
static int8_t rand_int8(void) { return (rand() % 31) - 15; } /* [-15,+15] */

ae32_t ae32_create(void)
{
    ae32_t m = {
        .W1 = tin_alloc(24*32),
        .W2 = tin_alloc(24*24),
        .W3 = tin_alloc(32*24),
        .h1 = tin_alloc(24),
        .h2 = tin_alloc(24),
        .y  = tin_alloc(32),
    };
    for(size_t i=0;i<m.W1.len;i++) m.W1.data[i]=rand_int8();
    for(size_t i=0;i<m.W2.len;i++) m.W2.data[i]=rand_int8();
    for(size_t i=0;i<m.W3.len;i++) m.W3.data[i]=rand_int8();
    return m;
}
void ae32_destroy(ae32_t* m)
{
    tin_free(&m->W1); tin_free(&m->W2); tin_free(&m->W3);
    tin_free(&m->h1); tin_free(&m->h2); tin_free(&m->y);
}

/* return integer‑scaled MSE (sum of squared error) */
uint32_t ae32_forward(ae32_t* m, const tensor_t* x)
{
    tin_dense_forward(&m->W1, x,       &m->h1);      /* 32→24 */
    tin_dense_forward(&m->W2, &m->h1,  &m->h2);      /* 24→24 */
    tin_dense_forward(&m->W3, &m->h2,  &m->y );      /* 24→32 */

    /* compute reconstruction error (int32) */
    uint32_t mse = 0;
    for(size_t i=0;i<32;i++){
        int16_t diff = (int16_t)x->data[i] - m->y.data[i];
        mse += diff*diff;              /* still small; fits in 32 b */
    }
    return mse;                        /* /32 deferred */
}

/* back‑prop: dense‑3 ➜ dense‑2 ➜ dense‑1 ------------------------------- */
void ae32_backward(ae32_t* m, const tensor_t* x)
{
    tensor_t err3 = tin_alloc(32);          /* error at output layer */
    for(size_t i=0;i<32;i++)
        err3.data[i] = clip_int8( x->data[i] - m->y.data[i] );   /* dMSE/dy */

    tensor_t err2 = tin_alloc(24);
    tin_dense_train(&m->W3, &m->h2, &err3, &err2);   /* update W3 */

    tensor_t err1 = tin_alloc(24);
    tin_dense_train(&m->W2, &m->h1, &err2, &err1);   /* update W2 */

    tensor_t dump = tin_alloc(32);                   /* not used further */
    tin_dense_train(&m->W1, x, &err1, &dump);        /* update W1 */

    tin_free(&dump); tin_free(&err1); tin_free(&err2); tin_free(&err3);
}
