#include<stdint.h>
/**
 * @file scale.h
 * @brief contains the header and apis for scale operations and definitions
 * @details this file contains scale structure, scale up / down count,
 * roll up, combine scales and scale shift
 * @author Shreyas Poyrekar
 * @date May 7, 2025
 * @license MIT
 */

typedef struct {
    int8_t S;   /* power‑of‑two shifts (+ = <<,  – = >>) */
    int8_t U;   /* # of up‑scales   (× 4⁄3) */
    int8_t D;   /* # of down‑scales (× 4⁄5) */
} _scale_t;

 /**
 * @class scale
 * @brief scale class contians Shift, Up, Down Counter
 *
 * For nested scaling the structure has a global and local counter
 * each for the shift, up and down
 *
 * @author Shreyas
 * @date 2025-05-07
 */
#ifdef TENSOR_USE_NESTED
typedef struct { _scale_t g, l; } scale_t;
#else
typedef _scale_t scale_t;
#endif

// scale operations
static inline void scale_combine(scale_t *dst, const scale_t *a, const scale_t *b);
static inline void scale_shift(scale_t *h, int8_t k);
static inline void scale_up(scale_t *h);
static inline void scale_down(scale_t *h);

#ifdef TENSOR_USE_NESTED
static inline void scale_rollup(scale_t *h);
#endif

