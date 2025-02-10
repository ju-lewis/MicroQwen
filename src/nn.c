
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

/* Performs a scaled dot product attention computation given queries, keys, and values */
Matrix scaled_dp_attention(Matrix *q, Matrix *k, Matrix *v, int requires_transpose) {
    
    // Deep clone matrix, in case keys are to be used later after the transpose
    Matrix keys_copy = clone_matrix(k);

    if (requires_transpose) {
        transpose(&keys_copy);
    }

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


Matrix grouped_query_attention(Matrix *x, Matrix *q_proj, Matrix *k_proj, Matrix *v_proj, Matrix *o_proj,
                                          Matrix *q_bias, Matrix *k_bias, Matrix *v_bias) {

    // First, compute the intial Q, K, V values (project into attention space)
    Matrix q_pre_bias = naive_matmul(x, q_proj);
    Matrix k_pre_bias = naive_matmul(x, k_proj);
    Matrix v_pre_bias = naive_matmul(x, v_proj);

    Matrix q = add_matrix(&q_pre_bias, q_bias); // n x 896
    Matrix k = add_matrix(&k_pre_bias, k_bias); // n x 128
    Matrix v = add_matrix(&v_pre_bias, v_bias); // n x 128

    
    // Partition based on the Q / KV head count
    MatrixPartition q_partitions[QUERY_HEAD_COUNT]; // Each partition is [n x 64]
    MatrixPartition k_partitions[KV_HEAD_COUNT];    // Each partition is [n x 64]
    MatrixPartition v_partitions[KV_HEAD_COUNT];    // Each partition is [n x 64]

    unsigned int  q_partition_size = q.n_cols / QUERY_HEAD_COUNT;
    unsigned int kv_partition_size = q.n_cols / KV_HEAD_COUNT;
    
    for (unsigned int i=0; i<QUERY_HEAD_COUNT; i++) {
        q_partitions[i] = partition_matrix(&q, q.n_rows, q_partition_size, 0, i*q_partition_size);
    }
    for (unsigned int i=0; i<KV_HEAD_COUNT; i++) {
        k_partitions[i] = partition_matrix(&k, q.n_rows, kv_partition_size, 0, i*kv_partition_size);
        v_partitions[i] = partition_matrix(&v, q.n_rows, kv_partition_size, 0, i*kv_partition_size);
    }

    // Compute all of the individual scaled dot product attention scores (7*2=14 total)
    Matrix attention_scores[QUERY_HEAD_COUNT];
    
    int score_idx = 0;
    for (int i=0; i<KV_HEAD_COUNT; i++) {
        for (int j=0; j<QUERY_HEAD_COUNT / KV_HEAD_COUNT; j++) {
            // Compute dot product attention for the current q/kv combination
            // The first 7 query heads correspond to the first KV head, and the
            // next 7 query heads correspond to the second KV head.
            attention_scores[score_idx] = scaled_dp_attention(&q_partitions[score_idx], &k_partitions[i], &v_partitions[i], false);

            score_idx++;
        }
    }

    // Concatenate the attention scores into an [n x 896] matrix (from 14 [n x 64] matrices)
    Matrix concat_output = concat_matrices(attention_scores, QUERY_HEAD_COUNT, COLUMN);

    // Project attention scores back to 'model space' with output proj matrix
    Matrix output = naive_matmul(&concat_output, o_proj);
    

    // That's a lot of heap memory lol
    for (int i=0; i<QUERY_HEAD_COUNT; i++) {
        free_matrix(&attention_scores[i]);
    }
    free_matrix(&concat_output);
    free_matrix(&q_pre_bias);
    free_matrix(&k_pre_bias);
    free_matrix(&v_pre_bias);
    free_matrix(&q);
    free_matrix(&k);
    free_matrix(&v);

    return output;
}



/* In-place ReLU activation function */
void relu(Matrix *logits) {
    // Ensure the Matrix represents a row vector
    assert(logits->n_rows == 1);

    // Apply ReLU activation across the whole vector
    for (unsigned int i=0; i<logits->n_cols; i++) {
        logits->vals[0][i] = MAX(0, logits->vals[0][i]);
    }
}

/* In-place softmax activation function */
void softmax(Matrix *logits) {
    // Ensure the Matrix represents a row vector
    assert(logits->n_rows == 1);

    // Sum exponentials of logits
    bfloat16 exp_sum = 0;
    for (unsigned int j=0; j<logits->n_cols; j++) {
        exp_sum = add_bf16(exp_sum, new_bf16(expf(bf16_to_float(logits->vals[0][j]))));
    }

    
    for (unsigned int i=0; i<logits->n_cols; i++) {
        logits->vals[0][i] = div_bf16(new_bf16(expf(bf16_to_float(logits->vals[0][i]))), exp_sum);
    }
}

/* In-place sigmoid activation function */
void sigmoid(Matrix *logits) {
    // Ensure the Matrix represents a row vector
    assert(logits->n_rows == 1);

    // Apply sigmoid activation across the whole vector
    for (unsigned int i=0; i<logits->n_cols; i++) {
        logits->vals[0][i] = new_bf16((float)(1.0f / (1.0f + exp(-bf16_to_float(logits->vals[0][i])))));
    }

}



