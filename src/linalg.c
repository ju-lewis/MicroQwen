
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>


#include "linalg.h"



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




