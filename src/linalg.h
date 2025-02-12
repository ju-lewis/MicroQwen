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

/* A partition into a `Matrix` object (Just provides semantic distinction) */
typedef Matrix MatrixPartition;


typedef enum {
    ROW,
    COLUMN
} Axis;

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

/* Perform a scalar multiplication over a matrix */
Matrix scalar_multiply(Matrix *m, bfloat16 coeff);

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

/* Deep-clones the values of a matrix */
Matrix clone_matrix(Matrix *m);

/* Creates a partition into a `Matrix` */
MatrixPartition partition_matrix(Matrix *m, unsigned int n_rows, unsigned int n_cols, 
        unsigned int row_offset, unsigned int col_offset);

/* concatenates `n` matrices along a given axis */
Matrix concat_matrices(Matrix *ms, int n, Axis axis);

/* In-place RMS Normalizes a vector, given a gain vector (for offsetting feature distribution) */
void rms_norm(Matrix *vec, Matrix *gain);


#define LINALG_H
#endif
