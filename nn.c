
#include "nn.h"


/* Creates an empty feed forward neural network */
FFModel init_ff_model() {
    return FFModel {
        .layers = NULL,
        .num_layers = 0
    };
}



