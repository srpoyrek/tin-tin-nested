
#ifndef TT_DENSE_H
#define TT_DENSE_H
#include "tt_types.h"

/* forward pass:  y = ReLU(W · x)  (Tin‑Tin scaling handled internally) */
void tt_dense_forward(const tensor_t *w,
                       const tensor_t *x,
                       tensor_t       *y);

#endif
