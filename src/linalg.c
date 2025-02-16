
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <math.h>


#include "linalg.h"

// Internal function declarations
Matrix *recursive_strassen(Matrix *a, Matrix *b);


/* =========================== BFLOAT FUNCTIONS ============================ */

bfloat16 new_bf16(float in) {
    bfloat16 x = 0;
    unsigned int s = sizeof(bfloat16);
    memcpy(
        &x,                         // Copy memory into the new bfloat16
        (float *)(((long)&in) + s), // Convert pointer to a `long` to allow shifting address to the sign and exponent
        s                           // Copy 2 bytes
    ); 
    return x;
}


float bf16_to_float(bfloat16 in) {
    float f = 0;
    unsigned int s = sizeof(bfloat16);
    memcpy(
        (float *)((long)(&f) + s), 
        &in, 
        s
    );
    return f;
}


/* Adds 2 bfloat16 values */
bfloat16 add_bf16(bfloat16 x, bfloat16 y) {
    float fx = bf16_to_float(x);
    float fy = bf16_to_float(y);
    return new_bf16(fx + fy);
}

/* Negates (changes the sign of) a bf16 */
bfloat16 negate_bf16(bfloat16 x) {
    return x ^= (1 << 15);
}

/* Multiplies 2 bfloat16 values */
bfloat16 mul_bf16(bfloat16 x, bfloat16 y) {
    float fx = bf16_to_float(x);
    float fy = bf16_to_float(y);
    return new_bf16(fx * fy);
}

/* Divides 2 bfloat16 values */
bfloat16 div_bf16(bfloat16 x, bfloat16 y) {
    float fx = bf16_to_float(x);
    float fy = bf16_to_float(y);
    return new_bf16(fx / fy);
}


/* =========================== MATRIX FUNCTIONS ============================ */

Matrix new_matrix(unsigned int n_rows, unsigned int n_cols) {

    bfloat16 **vals = (bfloat16 **)malloc(sizeof(bfloat16 *) * n_rows);
    assert(vals);

    for (unsigned int i=0; i<n_rows; i++) {
        vals[i] = (bfloat16 *)calloc(n_cols, sizeof(bfloat16));
        assert(vals[i]);
    }

    Matrix m = {
        .n_rows = n_rows,
        .n_cols = n_cols,
        .vals = vals
    };

    return m;
}


void print_matrix(Matrix *m) {
    for (unsigned int i=0; i<m->n_rows; i++) {
        for (unsigned int j=0; j<m->n_cols; j++) {
            printf("%f ", bf16_to_float(m->vals[i][j]));
        }
        putchar('\n');
    }
}

/* Perform simple addition on 2 matrices */
Matrix add_matrix(Matrix *a, Matrix *b) {
    assert(a->n_rows == b->n_rows);
    assert(a->n_cols == b->n_cols);

    Matrix c = new_matrix(a->n_rows, a->n_cols);

    for (unsigned int i=0; i < a->n_rows; i++) {
        for (unsigned int j=0; j < a->n_cols; j++) {
            c.vals[i][j] = add_bf16(a->vals[i][j], b->vals[i][j]);
        }
    }
    return c;
}

/* Perform simple subtraction on 2 matrices */
Matrix subtract_matrix(Matrix *a, Matrix *b) {
    assert(a->n_rows == b->n_rows);
    assert(a->n_cols == b->n_cols);

    Matrix c = new_matrix(a->n_rows, a->n_cols);

    for (unsigned int i=0; i < a->n_rows; i++) {
        for (unsigned int j=0; j < a->n_cols; j++) {
            c.vals[i][j] = add_bf16(a->vals[i][j], negate_bf16(b->vals[i][j]));
        }
    }
    return c;
}

/* Perform a scalar multiplication over a matrix */
Matrix scalar_multiply(Matrix *m, bfloat16 coeff) {
    Matrix result = new_matrix(m->n_rows, m->n_cols);

    for (unsigned int i=0; i<m->n_rows; i++) {
        for (unsigned int j=0; j<m->n_cols; j++) {
            result.vals[i][j] = mul_bf16(m->vals[i][j], coeff);
        }
    }

    return result;
}

Matrix naive_matmul(Matrix *a, Matrix *b) {
    assert(a->n_cols == b->n_rows);

    Matrix c = new_matrix(a->n_rows, b->n_cols);

    for (unsigned int i=0; i < a->n_rows; i++) {
        for (unsigned int k=0; k < a->n_cols; k++) {

            // Iterate through the relevant vector
            for(unsigned int j=0; j < b->n_cols; j++) {
                bfloat16 prod = mul_bf16(a->vals[i][k], b->vals[k][j]);
                c.vals[i][j] = add_bf16(c.vals[i][j], prod);
            }
        }
    }

    return c;
}

/* Prints the shape of a matrix to stdout */
void print_matrix_shape(Matrix *m) {
    printf("%dx%d\n", m->n_rows, m->n_cols);
}


/* Creates a one-hot encoding vector given a scalar quantity */
Matrix one_hot_encoding(unsigned int dim, unsigned int idx) {
    Matrix m = new_matrix(1, dim);

    m.vals[0][idx] = new_bf16(1);

    return m;
}



//Matrix strassen_matmul(Matrix *a, Matrix *b) {
//    assert(a->n_cols == b->n_rows);
//    
//    
//
//}
//


/* Computes the length of a row vector */
bfloat16 vector_magnitude(Matrix *vec) {
    assert(vec->n_rows = 1);
    
    bfloat16 sum = 0;
    for (unsigned int i=0; i<vec->n_cols; i++) {
        sum = add_bf16(sum, mul_bf16(vec->vals[0][i], vec->vals[0][i]));
    }

    // ewwwwwww
    return new_bf16((float)sqrt(bf16_to_float(sum)));
}


void pad_matrix(Matrix *m) {
    unsigned int target_size = MAX(m->n_rows, m->n_cols);
    

    unsigned int original_num_rows = m->n_rows;

    // Re-allocate rows
    if (m->n_rows < target_size) {
        m->vals = realloc(m->vals, target_size * sizeof(bfloat16 *));
        assert(m->vals);
        m->n_rows = target_size;
    }

    // Re-allocate columns
    for(unsigned int i=0; i<m->n_rows; i++) {
        m->vals[i] = realloc(m->vals[i], target_size * sizeof(bfloat16));
        assert(m->vals[i]);

        if (i < original_num_rows) {
            // Memset the rest of the row with 0
            memset(&m->vals[i][m->n_cols], 0, sizeof(bfloat16) * (target_size - m->n_cols));
        } else {
            memset(m->vals[i], 0, sizeof(bfloat16) * target_size);
        }
    }
    m->n_cols = target_size;
}


/* Frees the heap-allocated values in a `Matrix` */
void free_matrix(Matrix *m) {
    for (unsigned int i=0; i<m->n_rows; i++) {
        free(m->vals[i]);
    }
    free(m->vals);
}


/* In-place transposes a matrix */
void transpose(Matrix *m) {
    
    // Create new matrix to hold values
    Matrix mt = new_matrix(m->n_cols, m->n_rows);

    for (unsigned int i=0; i<m->n_rows; i++) {
        for (unsigned int j=0; j<m->n_cols; j++) {
            mt.vals[j][i] = m->vals[i][j];
        }
    }
    // Free the original values in the matrix
    free_matrix(m); 

    *m = mt;
}

/* Deep-clones the values of a matrix */
Matrix clone_matrix(Matrix *m) {
    Matrix new_m = new_matrix(m->n_rows, m->n_cols);
    
    for (unsigned int i=0; i<m->n_rows; i++) {
        memcpy(new_m.vals, m->vals, sizeof(bfloat16) * m->n_cols);
    }

    return new_m;
}

/* Creates a partition into a `Matrix` */
MatrixPartition partition_matrix(Matrix *m, unsigned int n_rows, unsigned int n_cols, unsigned int row_offset, unsigned int col_offset) {
    
    // Validate partition size
    assert(n_rows <= m->n_rows - row_offset && n_cols <= m->n_cols - col_offset);

    // Get row offset values
    bfloat16 **vals = &m->vals[row_offset];

    // Now iterate through and offset columns
    for (unsigned int i=0; i < n_rows; i++) {
        // Remember, C compiler treat this as syntactic sugar for vals[i] = &vals[i][col_offset]
        vals[i] += col_offset; 
    }


    MatrixPartition p = {
        .n_rows = n_rows,
        .n_cols = n_cols,
        .vals = vals
    };


    return p;
}


/* concatenates `n` matrices along a given axis */
Matrix concat_matrices(Matrix *ms, int n, Axis axis) {
    assert(n > 0);

    Matrix output;

    if (axis == COLUMN) {
        unsigned int total_cols = 0;
        for (int i=0; i<n; i++) {
            assert(ms[i].n_rows == ms[0].n_rows);
            total_cols += ms[i].n_cols;
        }
        
        output = new_matrix(ms[0].n_rows, total_cols);
        unsigned int cols_copied = 0;
        for (int i=0; i<n; i++) {

            // Copy each row from the current matrix
            for (unsigned int j=0; j<output.n_rows; j++) {
                memcpy(&output.vals[j][cols_copied], ms[i].vals[j], ms[i].n_cols);
            }

            cols_copied += ms[i].n_cols;
        }

    } else {
        unsigned int total_rows = 0;
        for (int i=0; i<n; i++) {
            assert(ms[i].n_cols == ms[0].n_cols);
            total_rows += ms[i].n_rows;
        }

        output = new_matrix(total_rows, ms[0].n_cols);
        unsigned int rows_copied = 0;
        for (int i=0; i<n; i++) {

            
            for (unsigned int j=0; j<ms[i].n_rows; j++) {
                // Copy individual row into output matrix
                memcpy(output.vals[rows_copied], ms[i].vals[j], ms[i].n_cols);
                rows_copied++;
            }
        }
    }
    return output;
}


/* In-place RMS Normalizes a vector, given a gain vector (for offsetting feature distribution) */
void rms_norm(Matrix *m, Matrix *gain) {

    // RMS normalise each row vector contained in the matrix
    for (unsigned int j=0; j<m->n_rows; j++) {

        float square_sum = 0;
        for (unsigned int i=0; i<m->n_cols; i++) {
            square_sum += powf(bf16_to_float(m->vals[j][i]), 2);
        }
        bfloat16 rms = new_bf16(sqrtf(square_sum / (float)m->n_cols));

        for (unsigned int i=0; i<m->n_cols; i++) {
            m->vals[j][i] = mul_bf16(div_bf16(m->vals[j][i], rms), gain->vals[j][i]);
        }
    }
}


/* Clones the `n`th row of an input Matrix (returning the resulting row vector) */
Matrix clone_nth_row(Matrix *m, unsigned int n) {
    assert(n < m->n_rows);

    Matrix vec = new_matrix(1, m->n_cols);
    memcpy(vec.vals, &m->vals[n], m->n_cols * sizeof(bfloat16));
    return vec;
}



