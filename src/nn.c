
#include <stdlib.h>
#include <assert.h>

#include "nn.h"



/* Creates an empty feed forward neural network */
FFModel init_ff_model() {

    FFModel m = {
        .layers = NULL,
        .num_layers = 0
    };

    return m;
}


void add_ff_layer(FFModel *model, Matrix *weights, Matrix (*activation_fn)(FFLayer *layer) ) {

    assert(weights != NULL);
    
    // Ensure the weights matrix is a correct size given the previous layer
    if (model->num_layers > 0) {
        assert(model->layers[model->num_layers - 1].weights.n_cols == weights->n_rows);
    }

    FFLayer new_layer = {
        .weights = *weights,
        .activation_fn = activation_fn
    };

    model->layers = (FFLayer *)realloc(model->layers, ++model->num_layers * sizeof(FFLayer));
    model->layers[model->num_layers - 1] = new_layer;
}


void free_ff_model(FFModel *model) {

    // Free the weights matrix for each layer
    for(size_t i=0; i<model->num_layers; i++) {
        free_matrix(&model->layers[i].weights);
    }

    free(model->layers);

}

/* Runs a full feed-forward prediction through a model */
Matrix ff_predict(FFModel *model, Matrix *input) {

    // Ensure the input is a row vector with the correct number of columns
    assert(
        model->num_layers > 0 && 
        input->n_rows == 1    && 
        model->layers[0].weights.n_rows == input->n_cols
    );
    
    // I'm going to start with a simple naive implementation that just creates and 
    // frees matrices for each layers' predictions.
    
    for(size_t i=0; i<model->num_layers; i++) {
    }

}





