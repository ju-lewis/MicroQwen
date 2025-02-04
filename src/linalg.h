/*
 * linalg.h defines all types and declares functions related to matrix operations
 * for MicroQwen
 *
 * ju-lewis 2025
 */
#ifndef LINALG_H
#include <stdint.h>


typedef uint16_t bfloat16;


typedef struct {
    unsigned int n_rows,
                 n_cols;
    bfloat16 **vals;
} Matrix;


/*
 * Creates a new uninitialized `Matrix` object of the given dimensions
 */
Matrix new_matrix(unsigned int n_rows, unsigned int n_cols);


/*
 * Pretty-prints a `Matrix` to stdout
 */
void print_matrix(Matrix *m);


#define LINALG_H
#endif
