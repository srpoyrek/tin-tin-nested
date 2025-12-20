#ifndef TT_DENSE_H
#define TT_DENSE_H
#include "tt_types.h"

/**
 * @brief Forward pass for dense layer: y = ReLU(W · x)
 * @param W Weight tensor (2D flattened as 1D)
 * @param X Input activation tensor
 * @param Y Output activation tensor
 * @param acc_buffer Accumulation buffer for intermediate calculations
 * @param acc_size Size of accumulation buffer (must equal Y->len)
 */
void tt_dense_forward(const tensor_t *W, const tensor_t *X, tensor_t *Y,
                      int32_t *acc_buffer, size_t acc_size);

/**
 * @brief Training pass for dense layer with backward propagation
 * @param W Weight tensor (updated in-place)
 * @param X Input activation tensor
 * @param err_next Error signal from next layer
 * @param err_prev Error signal to previous layer (output)
 * @param G_buffer Gradient buffer for weight updates
 */
void tt_dense_train(tensor_t *W, const tensor_t *X, const tensor_t *err_next,
                    tensor_t *err_prev, tensor_t *G_buffer);

#endif // TT_DENSE_H
