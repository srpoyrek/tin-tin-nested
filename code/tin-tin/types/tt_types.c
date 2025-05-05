#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tt_types.h"
#include "tt_debug.h"

/*
* init the tin-tin tensor
*/
void tt_tensor_init(tensor_t *t, int8_t *data, size_t len) {
    if(!t) return;
    if(!data) return;
    if(!len) return;
    t->data = data;
    t->len = len;
    t->s.S = 0;
    t->s.U = 0;
    t->s.D = 0;
    return;
}

/*
* reset or clear the tin-tin tensor
*/
void tt_tensor_clear(tensor_t *t) {
    if(!t) return;
    t->data = NULL;
    t->len = 0;
    t->s.S = 0;
    t->s.U = 0;
    t->s.D = 0;
    return;
}

/*
* print the tin-tin tensor
*/
#if TT_BIG_MACHINE_DEBUG_ENABLE
void tt_tensor_print(const char *name, const tensor_t *t)
{
    if(!t) {
        printf("%s tensor invalid!", name);
        return;
    }
    if(!t->data || t->len == 0) {
        printf("%s tensor empty!", name);
        return;
    }
    printf("%s  len=%zu  header<S,U,D>=(%d,%d,%d)\n",
           name, t->len, t->s.S, t->s.U, t->s.D);
    size_t n = t->len < 8 ? t->len : 8;
    printf("  data[0:%zu] =", t->len);
    for (size_t i = 0; i < n; ++i) printf(" %d", t->data[i]);
    puts(t->len > n ? " …" : "");
    return;
}
#endif