#include "tt_dense.h"
#include "tt_math.h"
#include <activations.h>
#include <stdlib.h>

/* identical body you wrote before, just inside the new file */
void tt_dense_forward(const tensor_t *w,
                       const tensor_t *x,
                       tensor_t       *y)
{
    size_t OUT = y->len, IN = x->len;
    int32_t *acc = malloc(OUT*sizeof(int32_t));

    for(size_t r=0; r<OUT; r++) {
        int32_t s = 0;
        for(size_t c = 0; c < IN; c++) {
            s += (int32_t)w->data[r*IN+c]*(int32_t)x->data[c];
        }
        acc[r] = relu_i8(s);
    }

    uint8_t bw   = eff_bitwidth_array(acc,OUT);
    uint8_t ksh  = (bw>8)?(bw-8):0;
    int8_t maxv=0;
    for(size_t r=0;r<OUT;r++){
        int8_t v = clip_int8( shift_and_round32(acc[r],ksh) );
        y->data[r]=v;
        if(abs(v)>maxv) maxv=abs(v);
    }
    y->s.S = w->s.S + x->s.S - ksh;
    y->s.U = w->s.U + x->s.U;
    y->s.D = w->s.D + x->s.D;

    if(maxv<32){
        for(size_t r=0;r<OUT;r++) y->data[r]=upscale_4_3(y->data[r]);
        y->s.U++;
    }else if(maxv>112){
        for(size_t r=0;r<OUT;r++) y->data[r]=downscale_4_5(y->data[r]);
        y->s.D++;
    }
    free(acc);
}
