/*
 * Defines types and declares functions relating to all neural network functionality
*/

#ifndef NN_H
#define NN_H

#include <stddef.h>
#include "linalg.h"


#define QWEN25_VOCAB_SIZE 151936
#define QUERY_HEAD_COUNT 14
#define KV_HEAD_COUNT 2
#define ROPE_THETA 10000.0


/* =================== FEED FORWARD NETWORK DECLARATIONS =================== */

typedef struct {
    Matrix weights;
    void (*activation_fn)(Matrix *pre_activation);
} FFLayer;


typedef struct {
    FFLayer *layers;
    size_t num_layers;
} FFModel;


/* Creates an empty feed forward neural network */
FFModel init_ff_model();

/* Adds a layer to a feed-forward neural network. */
void add_ff_layer(FFModel *model, Matrix *weights, void (*activation_fn)(Matrix *pre_activation));

/* Frees a feed-forward neural network */
void free_ff_model(FFModel *model);

/* Runs a full feed-forward prediction through a model */
Matrix ff_predict(FFModel *model, Matrix *input);


/* ==================== ATTENTION SUBLAYER DECLARATIONS ==================== */

/* Performs a scaled dot product attention computation given queries, keys, and values */
Matrix scaled_dp_attention(Matrix *q, Matrix *k, Matrix *v, int requires_transpose);

Matrix grouped_query_attention(Matrix *x, Matrix *q_proj, Matrix *k_proj, Matrix *v_proj, Matrix *o_proj,
                                          Matrix *q_bias, Matrix *k_bias, Matrix *v_bias);


/* ========================== ACTIVATION FUNCTIONS ========================= */

/* In-place ReLU activation function */
void relu(Matrix *logits);

/* In-place softmax activation function */
void softmax(Matrix *logits);

/* In-place sigmoid activation function */
void sigmoid(Matrix *logits);

/* In-place SiLU activation function */
void silu(Matrix *logits);

/* SwiGLU activation function, with Swish = SiLU (\beta = 1) */
Matrix swiglu(Matrix *logits, Matrix *up_proj, Matrix *gate_proj);


/* ============================= OTHER FUNCTIONS =========================== */

Matrix rotary_position_embedding(Matrix *vec, int position);

#endif
