
#ifndef TT_DENSE_H
#define TT_DENSE_H
#include "tt_types.h"

/* forward pass:  y = ReLU(W · x)  (Tin‑Tin scaling handled internally) */
void tt_dense_forward( const tensor_t *w, const tensor_t *x, tensor_t *y, int32_t *acc_buf);
void tt_dense_train(tensor_t *w, const tensor_t *x, const tensor_t *error_next, tensor_t *error_prev, tensor_t *buffer);  /* output */

#endif
