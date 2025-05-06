#ifndef TT_TYPES_H
#define TT_TYPES_H
#include <stdint.h>
#include <stddef.h>

/* -------- Tin‑Tin “classic” 3‑byte header -------------------- */
typedef struct {
    int8_t S;   /* power‑of‑two shifts (+ = <<,  – = >>) */
    int8_t U;   /* # of up‑scales   (× 4⁄3) */
    int8_t D;   /* # of down‑scales (× 4⁄5) */
} _scale_t;

#ifdef TENSOR_USE_NESTED
typedef struct { _scale_t g, l; } scale_t;
#else
typedef _scale_t scale_t;
#endif

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
