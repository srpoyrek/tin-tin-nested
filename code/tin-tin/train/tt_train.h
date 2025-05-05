#ifndef TT_TRAIN_H
#define TT_TRAIN_H
#include "tt_types.h"

/* outer‑product gradient + weight update (lr = 1/256) */
void tt_dense_train(tensor_t *w,
                     const tensor_t *x,
                     const tensor_t *error_next,
                     tensor_t *error_prev,
                     tensor_t *buffer);  /* output */

#endif
