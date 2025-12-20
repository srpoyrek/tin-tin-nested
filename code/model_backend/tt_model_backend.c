#include "tt_model_backend.h"
#include "motor_ae_model.h"

const ModelBackend_t tt_motor_model_backend = {
    .model_init = motor_ae_model_init,
    .model_forward  = motor_ae_model_forward,
    .model_backward = motor_ae_model_backward
};
