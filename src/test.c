
#include "linalg.h"

#include <stdio.h>


int main() {


    Matrix m = new_matrix(2, 2);

    m.vals[0][0] = new_bf16(1.0f);
    m.vals[1][1] = new_bf16(1.0f);


    print_matrix(&m);



}

