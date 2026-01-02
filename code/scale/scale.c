/**
 * @file scale.c
 * @author Shreyas Poyrekar
 * @date May 7, 2025
 */
#include "scale.h"


static void scale_init(scale_t *s)
{
    if (s != NULL)
    {
        s->S = s->U = s->D = 0;
    }

    return;
}

void scale_combine(scale_t *dst, const scale_t *a, const scale_t *b)
{
    if ((dst != NULL)
        && (a != NULL)
        && (b != NULL))
    {
        dst->S = a->S + b->S;
        dst->U = a->U + b->U;
        dst->D = a->D + b->D;
    }

    return;
}

void scale_shift(scale_t *h, int8_t k)
{
    if (h != NULL)
    {
        h->S += k;
    }

   return;
}

void scale_up(scale_t *h)
{
    if (h != NULL)
    {
        h->U++;
    }

    return;
}

void scale_down(scale_t *h)
{
    if (h != NULL)
    {
        h->D++;
    }
    return;
}

void scale_copy(scale_t *dst, const scale_t *src)
{
    if ((dst != NULL)
        && (src != NULL))
    {
        *dst = *src;
    }

    return;
}
