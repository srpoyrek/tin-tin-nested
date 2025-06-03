#ifndef TT_MODEL_BACKEND_H
#define TT_MODEL_BACKEND_H

#include "tt_tensor_backend.h"

// Backend vtable:
typedef struct {
    void (*model_init) (void *model, const TensorBackend_t *backend, uint32_t seed);
    void (*model_backward) (void *model, const tensor_t *x);
    uint32_t (*model_forward) (void *model, const tensor_t *x);    
} ModelBackend_t;

// Extern instances:
extern const ModelBackend_t tt_motor_model_backend;


#endif // MODEL_BACKEND_H