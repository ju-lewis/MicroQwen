
#include "nn.h"


/* Creates an empty feed forward neural network */
FFModel init_ff_model() {
    return FFModel {
        .layers = NULL,
        .num_layers = 0
    };
}


void add_ff_layer(FFModel *model) {
    model->layers = (FFLayer *)realloc(model->layers, ++model->num_layers * sizeof(FFLayer));
}



