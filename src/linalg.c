
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>


#include "linalg.h"



/* =========================== BFLOAT FUNCTIONS ============================ */

bfloat16 new_bf16(float in) {
    bfloat16 x;
    memcpy(
        &x,                         // Copy memory into the new bfloat16
        (float *)(((long)&in) + 2), // Convert pointer to a `long` to allow shifting address to the sign and exponent
        sizeof(bfloat16)            // Copy 2 bytes
    ); 
    return x;
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
            printf("%f ", (float)m->vals[i][j]);
        }
        putchar('\n');
    }
}




