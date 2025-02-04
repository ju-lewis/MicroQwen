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



/* =========================== BFLOAT FUNCTIONS ============================ */


/* Creates a new bfloat16 value from a float */
bfloat16 new_bf16(float in);

/* Converts a bfloat16 value to a float */
float bf16_to_float(bfloat16 in);

/* Adds 2 bfloat16 values */
bfloat16 add_bf16(bfloat16 x, bfloat16 y);

/* Multiplies 2 bfloat16 values */
bfloat16 mul_bf16(bfloat16 x, bfloat16 y);

/* Divides 2 bfloat16 values */
bfloat16 div_bf16(bfloat16 x, bfloat16 y);


/* =========================== MATRIX FUNCTIONS ============================ */

/* Creates a new uninitialized `Matrix` object of the given dimensions */
Matrix new_matrix(unsigned int n_rows, unsigned int n_cols);

/* Pretty-prints a `Matrix` to stdout */
void print_matrix(Matrix *m);

/* Naive (by-hand method) matrix multiplication - just for simple debugging */
Matrix naive_matmul(Matrix *a, Matrix *b);

/* Frees the heap-allocated values in a `Matrix` */
void free_matrix(Matrix *m);

#define LINALG_H
#endif
