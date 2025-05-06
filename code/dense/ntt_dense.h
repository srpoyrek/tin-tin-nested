
#ifndef NTT_DENSE_H
#define NTT_DENSE_H
#include "tt_types.h"

#ifdef TENSOR_USE_NESTED
/* forward pass:  y = ReLU(W · x)  (Tin‑Tin scaling handled internally) */
void ntt_dense_forward(const tensor_t *w,
                       const tensor_t *x,
                       tensor_t       *y);

void ntt_dense_train(tensor_t *w,
                    const tensor_t *x,
                    const tensor_t *error_next,
                    tensor_t *error_prev,
                    tensor_t *buffer);  /* output */

#endif
#endif