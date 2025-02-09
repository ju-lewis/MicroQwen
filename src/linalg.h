/*
 * linalg.h defines all types and declares functions related to matrix operations
 * for MicroQwen
 *
 * ju-lewis 2025
 */
#ifndef LINALG_H
#include <stdint.h>


#define MAX(x,y) x > y ? x : y


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

/* Negates (changes the sign of) a bf16 */
bfloat16 negate_bf16(bfloat16 x);

/* Multiplies 2 bfloat16 values */
bfloat16 mul_bf16(bfloat16 x, bfloat16 y);

/* Divides 2 bfloat16 values */
bfloat16 div_bf16(bfloat16 x, bfloat16 y);


/* =========================== MATRIX FUNCTIONS ============================ */


void pad_matrix(Matrix *m);

/* Creates a new uninitialized `Matrix` object of the given dimensions */
Matrix new_matrix(unsigned int n_rows, unsigned int n_cols);

/* Pretty-prints a `Matrix` to stdout */
void print_matrix(Matrix *m);

/* Perform simple addition on 2 matrices */
Matrix add_matrix(Matrix *a, Matrix *b);

/* Perform simple subtraction on 2 matrices */
Matrix subtract_matrix(Matrix *a, Matrix *b);

/* Naive (by-hand method) matrix multiplication - just for simple debugging */
Matrix naive_matmul(Matrix *a, Matrix *b);

/* Strassen (1969)'s matrix multiplication method */
Matrix strassen_matmul(Matrix *a, Matrix *b);

/* Computes the length of a row vector */
bfloat16 vector_magnitude(Matrix *vec);


/* Frees the heap-allocated values in a `Matrix` */
void free_matrix(Matrix *m);

/* Prints the shape of a matrix to stdout */
void print_matrix_shape(Matrix *m);

/* Creates a one-hot encoding vector given a scalar quantity */
Matrix one_hot_encoding(unsigned int dim, unsigned int idx);


/* In-place transposes a matrix */
void transpose(Matrix *m);



#define LINALG_H
#endif
