/*
 * Defines types and declares functions relating to all neural network functionality
*/

#ifndef NN_H
#define NN_H

#include <stddef.h>

#include "linalg.h"


typedef struct {
    Matrix weights;
    Matrix (*activation_fn)(FFLayer *layer)
} FFLayer;


typedef struct {
    FFLayer *layers;
    size_t num_layers;
} FFModel;


/* Creates an empty feed forward neural network */
FFModel init_ff_model();

//void add_ff_layer(FFModel *model);



#endif
