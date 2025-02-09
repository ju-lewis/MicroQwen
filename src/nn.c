
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "nn.h"



/* Creates an empty feed forward neural network */
FFModel init_ff_model() {

    FFModel m = {
        .layers = NULL,
        .num_layers = 0
    };

    return m;
}


void add_ff_layer(FFModel *model, Matrix *weights, void (*activation_fn)(Matrix *pre_activation)) {

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

    Matrix layer_inputs = *input;
    Matrix layer_outputs;
    
    for(size_t i=0; i<model->num_layers; i++) {
        layer_outputs = naive_matmul(&layer_inputs, &model->layers[i].weights);

        // Free previous inputs (as long as they weren't provided by the caller)
        // We can tell this if the underlying weight pointers are the same
        if (layer_inputs.vals != input->vals) {
            free_matrix(&layer_inputs);
        }

        // Modify the outputs in-place to apply the activation function
        model->layers[i].activation_fn(&layer_outputs);
        
        layer_inputs = layer_outputs;
    }

    
    return layer_outputs;

}


Matrix scaled_dp_attention(Matrix *q, Matrix *k, Matrix *v) {
    
    // Deep clone matrix, in case keys are to be used later after the transpose
    Matrix keys_copy = clone_matrix(k);

    transpose(&keys_copy);

    Matrix dot_prod = naive_matmul(q, &keys_copy);

    // Scale by the inverse square root of the dimension of the keys vector
    bfloat16 scale_factor = new_bf16((float)sqrt(k->n_cols));
    Matrix scaled_dot_prod = scalar_multiply(&dot_prod, scale_factor);
    
    
    softmax(&scaled_dot_prod);

    Matrix attention_result = naive_matmul(&scaled_dot_prod, v);


    free_matrix(&dot_prod);
    free_matrix(&keys_copy);
    free_matrix(&scaled_dot_prod);

    return attention_result;
}


void relu(Matrix *logits) {
    // Ensure the Matrix represents a row vector
    assert(logits->n_rows == 1);

    // Apply ReLU activation across the whole vector
    for (unsigned int i=0; i<logits->n_cols; i++) {
        logits->vals[0][i] = MAX(0, logits->vals[0][i]);
    }
}


void softmax(Matrix *logits) {
    // Ensure the Matrix represents a row vector
    assert(logits->n_rows == 1);

    // Sum exponentials of logits
    bfloat16 exp_sum = 0;
    for (unsigned int j=0; j<logits->n_cols; j++) {
        exp_sum = add_bf16(exp_sum, new_bf16(expf(bf16_to_float(logits->vals[0][j]))));
    }

    
    for (unsigned int i=0; i<logits->n_cols; i++) {
        logits->vals[0][i] = div_bf16(logits->vals[0][i], exp_sum);
    }

}



