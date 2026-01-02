/**
 * @file tt_types.c
 * @author Shreyas Poyrekar
 * @date May 7, 2025
 */

#include "tt_types.h"
#include <stdio.h>

void tt_tensor_init(tensor_t *t, tensor_shape_t shape, void *data, size_t len)
{
    if (!t || !data || !len) return;

    t->data = data;
    t->len = len;
    t->shape = shape;
    scale_init(&t->s);

    return;
}

void tt_tensor_clear(tensor_t *t) {
    if (!t) return;

    t->data = NULL;
    t->len = 0;

    scale_init(&t->s);

    t->shape.ndim = 0;
    for (size_t i = 0; i < 4; i++)
    {
        t->shape.shape[i] = 0;
    }

    return;
}


tensor_shape_t tt_tensor_get_shape(const tensor_t * const t)
{
    tensor_shape_t retval = {0};

    if (t != NULL)
    {
        retval = t->shape;
    }

    return retval;
}

#ifdef TT_BIG_MACHINE_DEBUG_ENABLE
void tt_tensor_print(const char *name, const tensor_t *t) {
    if (!t) {
        printf("%s: tensor invalid!\n", name);
        return;
    }
    if (!t->data || t->len == 0) {
        printf("%s: tensor empty!\n", name);
        return;
    }

#ifdef TENSOR_USE_NESTED
    printf("%s len=%zu scale: g(S=%d,U=%d,D=%d) l(S=%d,U=%d,D=%d)\n",
           name, t->len,
           t->s.g.S, t->s.g.U, t->s.g.D,
           t->s.l.S, t->s.l.U, t->s.l.D);
#else
    printf("%s len=%zu scale: (S=%d,U=%d,D=%d)\n",
           name, t->len, t->s.S, t->s.U, t->s.D);
#endif

    size_t n = (t->len < 8) ? t->len : 8;
    printf("  data[0:%zu] =", t->len);

    // Cast to int8_t for signed display (works for weights/gradients)
    int8_t *i8_data = (int8_t *)t->data;
    for (size_t i = 0; i < n; i++) {
        printf(" %d", i8_data[i]);
    }

    if (t->len > n) printf(" ...");
    printf("\n");
}
#endif
