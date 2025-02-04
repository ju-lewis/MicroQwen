
#include "linalg.h"

#include <stdio.h>


int main() {


    Matrix m = new_matrix(2, 2);

    m.vals[0][0] = new_bf16(2.0f);
    m.vals[0][1] = new_bf16(2.0f);

    m.vals[1][0] = new_bf16(1.0f);
    m.vals[1][1] = new_bf16(1.0f);


    Matrix vec = new_matrix(1, 2);

    vec.vals[0][0] = new_bf16(1.0f);
    vec.vals[0][1] = 0;
    

    Matrix scaled_vec = naive_matmul(&vec, &m);

    printf("Before: \n");
    print_matrix(&vec);

    printf("After: \n");
    print_matrix(&scaled_vec);





}

