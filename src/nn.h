/*
 * Defines types and declares functions relating to all neural network functionality
*/

#ifndef NN_H
#define NN_H

#include <stddef.h>
#include "linalg.h"


#define QWEN25_VOCAB_SIZE 151936


struct FFLayer {
    Matrix weights;
    Matrix (*activation_fn)(struct FFLayer *layer);
};

typedef struct FFLayer FFLayer;


typedef struct {
    FFLayer *layers;
    size_t num_layers;
} FFModel;


/* Creates an empty feed forward neural network */
FFModel init_ff_model();

/* Adds a layer to a feed-forward neural network */
void add_ff_layer(FFModel *model);



#endif
