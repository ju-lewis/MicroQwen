
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <math.h>


#include "linalg.h"

// Internal function declarations
Matrix *recursive_strassen(Matrix *a, Matrix *b);
unsigned int next_largest_2_power(unsigned int n);


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


bfloat16 add_bf16(bfloat16 x, bfloat16 y) {
    float fx = bf16_to_float(x);
    float fy = bf16_to_float(y);
    return new_bf16(fx + fy);
}


bfloat16 mul_bf16(bfloat16 x, bfloat16 y) {
    float fx = bf16_to_float(x);
    float fy = bf16_to_float(y);
    return new_bf16(fx * fy);
}


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
        vals[i] = (bfloat16 *)malloc(sizeof(bfloat16) * n_cols);
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


Matrix naive_matmul(Matrix *a, Matrix *b) {
    assert(a->n_cols == b->n_rows);

    Matrix c = new_matrix(a->n_rows, b->n_cols);

    for (unsigned int i=0; i < a->n_rows; i++) {
        for(unsigned int j=0; j < b->n_cols; j++) {

            bfloat16 curr = 0;

            // Iterate through the relevant vector
            for (unsigned int k=0; k < a->n_cols; k++) {
                bfloat16 prod = mul_bf16(a->vals[i][k], b->vals[k][j]);
                curr = add_bf16(curr, prod);
            }

            c.vals[i][j] = curr;
        }
    }

    return c;
}


//Matrix strassen_matmul(Matrix *a, Matrix *b) {
//    assert(a->n_cols == b->n_rows);
//    
//    
//
//}
//
void power_2_pad_matrix(Matrix *m) {
    unsigned int largest_dimension = MAX(m->n_rows, m->n_cols);
    
    unsigned int target_size = next_largest_2_power(largest_dimension);

    printf("Target size: %dx%d\n", target_size, target_size);
    
    
    // Re-allocate rows
    if (m->n_rows < target_size) {

    }

    // Re-allocate columns
    if (m->n_cols < target_size) {
        
    }
}



unsigned int next_largest_2_power(unsigned int n) {
    // Check if already a power of 2
    if ((n & (n-1)) == 0) {
        return n;
    }
    return 2 << (int)log2(n);

}


void free_matrix(Matrix *m) {
    free(m->vals);
    m->vals = NULL;
}



