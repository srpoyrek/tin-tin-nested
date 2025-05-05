#ifndef TIN_AE_H
#define TIN_AE_H
#include "tin_types.h"

typedef struct {
    /* weight tensors (row‑major) */
    tensor_t W1;   /* 24 × 32 */
    tensor_t W2;   /* 24 × 24 */
    tensor_t W3;   /* 32 × 24 */
    /* activations reused during back‑prop */
    tensor_t h1, h2, y;    /* size 24, 24, 32 */
} ae32_t;

/* allocate + random‑init */
ae32_t ae32_create(void);
/* free all buffers */
void   ae32_destroy(ae32_t*);
/* forward; returns recon‑error (MSE×256 for integer convenience) */
uint32_t ae32_forward(ae32_t*, const tensor_t* x);
/* backward pass + weight update  (SGD lr = 1/256) */
void   ae32_backward(ae32_t*, const tensor_t* x);

#endif
