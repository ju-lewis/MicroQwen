
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <string.h>

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
        // NOTE: Qwen2.5 doesn't apply any biases in the dense feed forward network, only QKV bias in the GQA attention sublayer
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
        assert(q->n_cols == k->n_cols && k->n_cols == v->n_cols);
    } else {
        assert(q->n_rows == k->n_rows && k->n_rows == v->n_rows);
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

    // We need to add the bias to each row of the matrix (in case of batch processing 
    // embedding row vectors) to allow us to add the bias
    Matrix q_bias_padded = new_matrix(x->n_rows, q_bias->n_cols);
    Matrix k_bias_padded = new_matrix(x->n_rows, k_bias->n_cols);
    Matrix v_bias_padded = new_matrix(x->n_rows, v_bias->n_cols);
    for (unsigned int i=0; i<x->n_rows; i++) {
        memcpy(q_bias_padded.vals[i], q_bias->vals[0], q_bias->n_cols);
        memcpy(k_bias_padded.vals[i], k_bias->vals[0], k_bias->n_cols);
        memcpy(v_bias_padded.vals[i], v_bias->vals[0], v_bias->n_cols);
    }
    Matrix q = add_matrix(&q_pre_bias, &q_bias_padded); // n x 896
    Matrix k = add_matrix(&k_pre_bias, &k_bias_padded); // n x 128
    Matrix v = add_matrix(&v_pre_bias, &v_bias_padded); // n x 128

    
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
            attention_scores[score_idx] = scaled_dp_attention(&q_partitions[score_idx], &k_partitions[i], &v_partitions[i], 0);

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
    free_matrix(&q_bias_padded);
    free_matrix(&k_bias_padded);
    free_matrix(&v_bias_padded);
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

    for (unsigned int r=0; r<logits->n_rows; r++) {

        // Sum exponentials of logits
        bfloat16 exp_sum = 0;
        for (unsigned int j=0; j<logits->n_cols; j++) {
            exp_sum = add_bf16(exp_sum, new_bf16(expf(bf16_to_float(logits->vals[r][j]))));
        }
        
        for (unsigned int i=0; i<logits->n_cols; i++) {
            logits->vals[r][i] = div_bf16(new_bf16(expf(bf16_to_float(logits->vals[r][i]))), exp_sum);
        }
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

/* In-place SiLU activation function */
void silu(Matrix *logits) {
    assert(logits->n_rows == 1);

    // Apply SiLU across the whole vector
    for (unsigned int i=0; i<logits->n_cols; i++) {
        logits->vals[0][i] = div_bf16(
            logits->vals[0][i], 
            new_bf16(1.0f + expf(-bf16_to_float(logits->vals[0][i])))
        );
    }
}


/* SwiGLU activation function, with Swish = SiLU (\beta = 1) */
Matrix swiglu(Matrix *logits, Matrix *up_proj, Matrix *gate_proj) {

    // Project input vector
    Matrix a = naive_matmul(logits, up_proj);
    Matrix b = naive_matmul(logits, gate_proj);
    assert(a.n_rows == b.n_rows && a.n_cols == b.n_cols);

    // Run Swish (SiLU in this case) on the gate projection
    silu(&b);

    // Perform an element-wise product between a and b
    Matrix output = new_matrix(a.n_rows, a.n_cols);
    for (unsigned int i=0; i<a.n_cols; i++) {
        output.vals[0][i] = mul_bf16(a.vals[0][i], b.vals[0][i]);
    }

    free_matrix(&a);
    free_matrix(&b);

    return output;
}


/* Performs a rotary position embedding (RoPE) on an input vector */
Matrix rotary_position_embedding(Matrix *vec, int position) {
    
    // Assert `vec` is an even-width column vector
    assert(vec->n_rows == 1 && vec->n_cols % 2 == 0);

    float rotation = ROPE_THETA * (float)position;
    
    Matrix rotation_matrix = new_matrix(vec->n_cols, vec->n_cols);
    // Increment by 2
    for (unsigned int i=0; i<vec->n_cols; i += 2) {
        // Create 2x2 (2D) rotation matrix in the current diagonal block

        vec->vals[i][i] = new_bf16(cosf(rotation));
        vec->vals[i+1][i+1] = new_bf16(cosf(rotation));

        vec->vals[i][i] = new_bf16(sinf(rotation));
        vec->vals[i+1][i+1] = new_bf16(-sinf(rotation));
    }
    
    // Apply 2D pair-wise rotations to the input vector
    Matrix output = naive_matmul(&rotation_matrix, vec);
    return output;
}

/* Predicts the next embedding vector given a `Decoder` model and the previous 
 * sequence of position encoded embeddings, as a [seq_len x d_model] `Matrix`.
*/
Matrix predict_next_embedding(Decoder *model, Matrix *sequence) {

    assert(sequence->n_cols == model->d_model);

    Matrix curr_input = *sequence, 
           prev_input = *sequence;
    
    // Iterate through all decoder blocks
    for (unsigned int i=0; i<model->n_layers; i++) {
        // Shorten names
        TransformerCell c = model->layers[i];
        AttentionLayer attn_l = c.attn;

        // Grouped Query Attention layer
        Matrix attn_result = grouped_query_attention(sequence, &attn_l.q_proj, &attn_l.k_proj, &attn_l.v_proj, &attn_l.o_proj,
                                                               &attn_l.q_bias, &attn_l.k_bias, &attn_l.v_bias);

        // Residual connection and normalisation
        Matrix after_attn_residuals = add_matrix(&attn_result, sequence);
        rms_norm(&after_attn_residuals, &c.input_norm); 


        // FFN
        Matrix ffn_result = ff_predict(&c.ffn, &after_attn_residuals);

        // Residual connection and normalisation
        Matrix after_ffn_residuals = add_matrix(&ffn_result, &after_attn_residuals);
        rms_norm(&after_ffn_residuals, &c.output_norm);

        // Free all unused matrices
        

        // Free all input matrices except for the initial input
        if (prev_input.vals != sequence->vals) {
            free_matrix(&prev_input);
        }
        prev_input = curr_input;
        curr_input = after_ffn_residuals;
    }

    // The predicted next embedding vector in the sequence will be the *last* row of the output `Matrix`
    Matrix next_embedding = clone_nth_row(&curr_input, curr_input.n_rows - 1);
    
    return next_embedding;
}

