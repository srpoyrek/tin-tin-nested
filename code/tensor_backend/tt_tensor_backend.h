#ifndef TT_TENSOR_BACKEND_H
#define TT_TENSOR_BACKEND_H

#include "tt_types.h"

// Backend vtable:
typedef struct {
    void (*dense_forward)(const tensor_t*, const tensor_t*, tensor_t*, int32_t*, size_t);
    void (*dense_train)(tensor_t*, const tensor_t*, const tensor_t*, tensor_t*, tensor_t*);
} TensorBackend_t;

// Extern instances:
extern const TensorBackend_t tt_backend;

#ifdef TENSOR_USE_NESTED
extern const TensorBackend_t nested_backend;
#endif

#endif // TENSOR_BACKEND_H