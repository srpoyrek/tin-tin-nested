#include "ntt_dense.h"
#include "tt_math.h"    // shift_and_round32, upscale_4_3, downscale_4_5, eff_bitwidth_array
#include <stddef.h>

#ifdef TENSOR_USE_NESTED

/* roll-up limits for nested header */
#define NTT_ROLL_LIM   16
#define NTT_ROLL_STEP   8
#define NTT_LR_SHIFT    8    /* lr = 1/256 */
#define NTT_MARGIN      2    /* Algorithm 3 margin */

/* ---------- nested header utilities --------------------------- */
static inline void roll_up(scale_t *h) {
    /* local S */
    if (h->l.S >  NTT_ROLL_LIM) { h->g.S += NTT_ROLL_STEP; h->l.S -= NTT_ROLL_STEP; }
    if (h->l.S < -NTT_ROLL_LIM) { h->g.S -= NTT_ROLL_STEP; h->l.S += NTT_ROLL_STEP; }
    /* local U */
    if (h->l.U >  NTT_ROLL_LIM) { h->g.U += NTT_ROLL_STEP; h->l.U -= NTT_ROLL_STEP; }
    /* local D */
    if (h->l.D >  NTT_ROLL_LIM) { h->g.D += NTT_ROLL_STEP; h->l.D -= NTT_ROLL_STEP; }
}

static void align_global(scale_t *w, scale_t *g) {
    /* align global counters only (no data moves) */
    int8_t dS = w->g.S - g->g.S;
    w->g.S -= dS;  g->l.S += dS;
    int8_t dU = w->g.U - g->g.U;
    w->g.U -= dU;  g->l.U += dU;
    int8_t dD = w->g.D - g->g.D;
    w->g.D -= dD;  g->l.D += dD;
}

static void align_local(tensor_t *W, tensor_t *G) {
    /* align local counters by shifting G.data accordingly */
    int8_t dS = W->s.l.S - G->s.l.S;
    if (dS > 0) {
        for (size_t i = 0; i < G->len; ++i)
            G->data[i] <<= dS;
        G->s.l.S += dS;
    } else if (dS < 0) {
        dS = -dS;
        for (size_t i = 0; i < G->len; ++i)
            G->data[i] = shift_and_round32(G->data[i], dS);
        G->s.l.S -= dS;
    }
    int8_t dU = W->s.l.U - G->s.l.U;
    while (dU-- > 0) {
        for (size_t i = 0; i < G->len; ++i)
            G->data[i] = upscale_4_3(G->data[i]);
        G->s.l.U++;
    }
    int8_t dD = W->s.l.D - G->s.l.D;
    while (dD-- > 0) {
        for (size_t i = 0; i < G->len; ++i)
            G->data[i] = downscale_4_5(G->data[i]);
        G->s.l.D++;
    }
}

/* ---------- nested forward ------------------------------------- */
void ntt_dense_forward(const tensor_t *w,
                       const tensor_t *x,
                       tensor_t       *y)
{
    size_t OUT = y->len;
    size_t IN  = x->len;
    int32_t acc[OUT];
    /* 1) dot-product + ReLU */
    for (size_t r = 0; r < OUT; ++r) {
        int32_t sum = 0;
        for (size_t c = 0; c < IN; ++c)
            sum += (int32_t)w->data[r*IN + c] * (int32_t)x->data[c];
        acc[r] = (sum < 0) ? 0 : sum;
    }
    /* 2) choose shift */
    uint8_t bw  = eff_bitwidth_array(acc, OUT);
    uint8_t ksh = (bw > 8) ? (bw - 8) : 0;
    /* 3) shrink to int8 and track max */
    int8_t maxv = 0;
    for (size_t r = 0; r < OUT; ++r) {
        int8_t v = clip_int8( shift_and_round32(acc[r], ksh) );
        y->data[r] = v;
        if ((int8_t)abs(v) > maxv) maxv = abs(v);
    }
    /* 4) nested header update */
    /* global parts accumulate */
    y->s.g.S = w->s.g.S + x->s.g.S;
    y->s.g.U = w->s.g.U + x->s.g.U;
    y->s.g.D = w->s.g.D + x->s.g.D;
    /* local parts accumulate, including shift */
    y->s.l.S = w->s.l.S + x->s.l.S - ksh;
    y->s.l.U = w->s.l.U + x->s.l.U;
    y->s.l.D = w->s.l.D + x->s.l.D;
    /* 5) up/downscale locally */
    if (maxv < 32) {
        for (size_t i = 0; i < OUT; ++i)
            y->data[i] = upscale_4_3(y->data[i]);
        y->s.l.U++;
    } else if (maxv > 112) {
        for (size_t i = 0; i < OUT; ++i)
            y->data[i] = downscale_4_5(y->data[i]);
        y->s.l.D++;
    }
    /* 6) roll-up to keep local bounded */
    roll_up(&y->s);
}

/* ---------- nested train (Alg 3) ------------------------------ */
void ntt_dense_train(tensor_t *W,
                     const tensor_t *x,
                     const tensor_t *err_next,
                     tensor_t *err_prev,
                     tensor_t *buffer)
{
    size_t OUT = err_next->len;
    size_t IN  = x->len;
    /* ensure buffer length */
    buffer->len = W->len;
    /* 1) outer-product gradient into buffer */
    for (size_t r = 0; r < OUT; ++r) {
        for (size_t c = 0; c < IN; ++c) {
            int16_t g16 = err_next->data[r] * x->data[c];
            buffer->data[r*IN + c] = clip_int8(g16);
        }
    }
    /* grad header sum global and local */
    buffer->s.g.S = W->s.g.S + x->s.g.S;
    buffer->s.g.U = W->s.g.U + x->s.g.U;
    buffer->s.g.D = W->s.g.D + x->s.g.D;
    buffer->s.l.S = W->s.l.S + x->s.l.S;
    buffer->s.l.U = W->s.l.U + x->s.l.U;
    buffer->s.l.D = W->s.l.D + x->s.l.D;
    /* 2) lr shift in local */
    for (size_t i = 0; i < buffer->len; ++i)
        buffer->data[i] = clip_int8( shift_and_round32(buffer->data[i], NTT_LR_SHIFT) );
    buffer->s.l.S -= NTT_LR_SHIFT;
    /* 3) align global then local */
    align_global(&W->s, &buffer->s);
    align_local(W, buffer);
    /* 4) margin-based bw adjust on buffer.local */
    {
        uint8_t b_g=0, b_w=0;
        for (size_t i=0;i<W->len;++i) {
            uint8_t bg = eff_bitwidth32(buffer->data[i]);
            uint8_t bw = eff_bitwidth32(W->data[i]);
            if(bg>b_g) b_g=bg;
            if(bw>b_w) b_w=bw;
        }
        int8_t target = (int8_t)b_w - NTT_MARGIN;
        int8_t shift = (int8_t)b_g - target;
        if (shift>0) {
            for (size_t i=0;i<buffer->len;++i)
                buffer->data[i] = clip_int8( shift_and_round32(buffer->data[i], shift) );
            buffer->s.l.S -= shift;
        }
    }
    /* 5) SGD update */
    for (size_t i = 0; i < W->len; ++i)
        W->data[i] = clip_int8(W->data[i] - buffer->data[i]);
    /* 6) optional weight renorm (local only) */
    int8_t maxw=0;
    for (size_t i=0;i<W->len;++i)
        if(abs(W->data[i])>maxw) maxw=abs(W->data[i]);
    if (maxw>112) {
        for(size_t i=0;i<W->len;++i)
            W->data[i] = downscale_4_5(W->data[i]);
        W->s.l.D++;
    } else if (maxw<32) {
        for(size_t i=0;i<W->len;++i)
            W->data[i] = upscale_4_3(W->data[i]);
        W->s.l.U++;
    }
    roll_up(&W->s);
    /* 7) backprop error */
    for (size_t c = 0; c < IN; ++c) {
        int32_t sum=0;
        for (size_t r = 0; r < OUT; ++r)
            sum += (int32_t)W->data[r*IN + c] * err_next->data[r];
        err_prev->data[c] = clip_int8( shift_and_round32(sum, 7) );
    }
    /* error header nested */
    err_prev->s.g.S = W->s.g.S + err_next->s.g.S;
    err_prev->s.g.U = W->s.g.U + err_next->s.g.U;
    err_prev->s.g.D = W->s.g.D + err_next->s.g.D;
    err_prev->s.l.S = W->s.l.S + err_next->s.l.S - 7;
    err_prev->s.l.U = W->s.l.U + err_next->s.l.U;
    err_prev->s.l.D = W->s.l.D + err_next->s.l.D;
    roll_up(&err_prev->s);
}

#endif