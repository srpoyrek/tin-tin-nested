#include "tt_dense.h"
#include "tt_math.h"
#include "matrix.h"
#include "activations.h"
#include <stdlib.h>
#include <string.h>

/* constants */
#define LR_SHIFT 8    /* lr = 1 / 256   */
#define MARGIN   2    /* Algorithm‑3 line 10 */

static void align_scale(tensor_t *W, tensor_t *G_buffer);
static inline void activation_function(int32_t * acc_buffer, size_t len, Activation_i8_t func);

/* identical body you wrote before, just inside the new file */
void tt_dense_forward(const tensor_t *W, const tensor_t *X, tensor_t *Y, int32_t * acc_buffer) {
    if(!W || !X || !Y || !acc_buffer) return;
    
    size_t OUT = Y->len, IN = X->len;
    
    // raw int32 matrix-vector multiplication
    matrix_mul(W, X, acc_buffer);

    // apply ReLU to the int32 accumulators
    activation_function(acc_buffer, OUT, relu_i8);

    uint8_t bw   = eff_bitwidth_array(acc_buffer,OUT);
    uint8_t ksh  = (bw > 8) ? ( bw - 8 ) : 0 ;
    int8_t maxv = 0;
    
    for(size_t r = 0; r < OUT; r++) {
        int8_t v = clip_int8( shift_and_round32(acc_buffer[r],ksh) );
        Y->data[r]=v;
        if(abs(v)>maxv) maxv=abs(v);
    }

    scale_combine(&Y->s, &W->s, &X->s);
    scale_shift(&Y->s, - (int8_t)ksh);

    if(maxv<32) {
        for(size_t r = 0; r < OUT; r++) {
            Y->data[r]=upscale_4_3(Y->data[r]);
        }
        Y->s.U++;
    } else if(maxv > 112) {
        for(size_t r = 0; r < OUT; r++) {
            Y->data[r]=downscale_4_5(Y->data[r]);
        }
        Y->s.D++;
    }
    return;
}

static inline void activation_function(int32_t *acc, size_t len, Activation_i8_t func) {
    if(!acc) return;
    for (size_t i = 0; i < len; ++i) {
        acc[i] = func(acc[i]);
    }
    return;
}

/* single‑header alignment (Alg 3 lines 1–6) */
static void align_scale(tensor_t *W, tensor_t *G_buffer)
{
    int8_t dS = W->s.S - G_buffer->s.S;
    int8_t dU = W->s.U - G_buffer->s.U;
    int8_t dD = W->s.D - G_buffer->s.D;

    if (dS > 0) {
        for (size_t i = 0; i < G_buffer->len; ++i)
            G_buffer->data[i] <<= dS;
        G_buffer->s.S += dS;
    } else if (dS < 0) {
        dS = -dS;
        for (size_t i = 0; i < G_buffer->len; ++i)
            G_buffer->data[i] = shift_and_round32(G_buffer->data[i], dS);
        G_buffer->s.S -= dS;
    }

    while (dU-- > 0) {
        for (size_t i = 0; i < G_buffer->len; ++i)
            G_buffer->data[i] = upscale_4_3(G_buffer->data[i]);
        G_buffer->s.U++;
    }
    while (dD-- > 0) {
        for (size_t i = 0; i < G_buffer->len; ++i)
            G_buffer->data[i] = downscale_4_5(G_buffer->data[i]);
        G_buffer->s.D++;
    }
}


void tt_dense_train(tensor_t *W,
                    const tensor_t *x,
                    const tensor_t *err_next,
                    tensor_t *err_prev,
                    tensor_t *G_buffer)
{
    size_t OUT = err_next->len;
    size_t IN  = x->len;

    /* make sure G_buffer has the right length */
    G_buffer->len = W->len;

    /* 1. gradient wrt weights: outer‑product ----------------------- */
    for (size_t r = 0; r < OUT; ++r) {
        for (size_t c = 0; c < IN; ++c) {
            int16_t g16 = (int16_t)err_next->data[r] * x->data[c];
            G_buffer->data[r*IN + c] = clip_int8(g16);
        }
    }
    G_buffer->s.S = err_next->s.S + x->s.S;
    G_buffer->s.U = err_next->s.U + x->s.U;
    G_buffer->s.D = err_next->s.D + x->s.D;

    /* 2. learning‑rate multiply (>>8) */
    for (size_t i = 0; i < G_buffer->len; ++i) {
        G_buffer->data[i] = clip_int8( shift_and_round32(G_buffer->data[i], LR_SHIFT) );
    }
    G_buffer->s.S -= LR_SHIFT;

    /* 3. align scales (Alg 3 lines 1–6) */
    align_scale(W, G_buffer);

    /* --- Alg 3 lines 8–11: margin bit‑width adjustment ---------- */
    {
        uint8_t b_g = 0, b_w = 0;
        /* compute bit‑width of gradient and weights */
        for (size_t i = 0; i < W->len; ++i) {
            int32_t gv = G_buffer->data[i];
            int32_t wv = W->data[i];
            uint8_t bg = eff_bitwidth32(gv);
            uint8_t bw = eff_bitwidth32(wv);
            if (bg > b_g) b_g = bg;
            if (bw > b_w) b_w = bw;
        }
        int8_t b = (int8_t)b_w - MARGIN;           /* target bit‑width */
        int8_t shift_adj = (int8_t)b_g - b;        /* how much to shift G */
        if (shift_adj > 0) {
            for (size_t i = 0; i < G_buffer->len; ++i) {
                G_buffer->data[i] = clip_int8(
                    shift_and_round32(G_buffer->data[i], shift_adj)
                );
            }
            G_buffer->s.S -= shift_adj;            /* update scale header */
        }
    }

    /* 4. SGD update ---------------------------- */
    for (size_t i = 0; i < W->len; ++i) {
        W->data[i] = clip_int8(W->data[i] - G_buffer->data[i]);
    }

    /* optional weight renorm ------------------ */
    int8_t maxw = 0;
    for (size_t i = 0; i < W->len; ++i)
        if (abs(W->data[i]) > maxw) maxw = abs(W->data[i]);

    if (maxw > 112) {
        for (size_t i = 0; i < W->len; ++i)
            W->data[i] = downscale_4_5(W->data[i]);
        W->s.D++;
    } else if (maxw < 32) {
        for (size_t i = 0; i < W->len; ++i)
            W->data[i] = upscale_4_3(W->data[i]);
        W->s.U++;
    }

    /* 5. error to previous layer: Wᵀ·err_next --------------------- */
    for (size_t c = 0; c < IN; ++c) {
        int32_t acc = 0;
        for (size_t r = 0; r < OUT; ++r) {
            acc += (int32_t)W->data[r*IN + c] * err_next->data[r];
        }
        err_prev->data[c] = clip_int8( shift_and_round32(acc, 7) );  /* quick */
    }
    err_prev->s.S = W->s.S + err_next->s.S - 7;
}
