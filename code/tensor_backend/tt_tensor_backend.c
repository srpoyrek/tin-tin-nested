#include "tt_tensor_backend.h"
#include "tt_dense.h"
#include "ntt_dense.h"

const TensorBackend_t tt_backend = {
    .dense_forward = tt_dense_forward,
    .dense_train   = tt_dense_train
};

#ifdef TENSOR_USE_NESTED
const TensorBackend_t nested_backend = {
    .dense_forward = ntt_dense_forward,
    .dense_train   = ntt_dense_train
};
#endif