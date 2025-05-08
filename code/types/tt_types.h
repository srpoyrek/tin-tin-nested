#ifndef TT_TYPES_H
#define TT_TYPES_H
#include"scale.h"

/* -------- 1‑D tensor ------------------- */
typedef struct {
    int8_t  *data;     /* int‑8 payload */
    size_t   len;
    scale_t  s;        /* scale header */
} tensor_t;

/* init */
void tt_tensor_init(tensor_t *t, int8_t *data, size_t len);
void tt_tensor_clear(tensor_t *t);

/* Debug print */
void tt_tensor_print(const char *name, const tensor_t *t);


#endif
