/**
 * @file scale.h
 * @brief contains the header and apis for scale operations and definitions
 * @details this file contains scale structure, scale up / down count,
 * roll up, combine scales and scale shift
 * @author Shreyas Poyrekar
 * @date May 7, 2025
 * @license MIT
 */
#include <stdint.h>

typedef struct
{
    int8_t S;   /* Power-of-two shifts: scale × 2^S */
    int8_t U;   /* Up-scales: scale × (4/5)^U */
    int8_t D;   /* Down-scales: scale × (3/4)^D */
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
void scale_combine(scale_t* dst, const scale_t* a, const scale_t* b);

/**
 * @brief shifts scales
 * @param h pointer of the scale to shift
 * @param k shift by
 */
void scale_shift(scale_t* h, int8_t k);

/**
 * @brief scales up
 * increments by one
 * @param h pointer of the scale to up
 */
void scale_up(scale_t* h);

/**
 * @brief scales down
 * decrements by one
 * @param h pointer of the scale to down
 */
void scale_down(scale_t* h);

/**
 * @brief copy scale
 * @param dst ptr to the scale to copy
 * @param src ptr to the scale to copy to.
 */
void scale_copy(scale_t* dst, const scale_t* src);

#ifdef TENSOR_USE_NESTED
void scale_rollup(scale_t* h);
#endif
