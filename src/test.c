
#include "linalg.h"

#include <stdio.h>


int main() {
    
    bfloat16 x = new_bf16(1.3);

    printf("%016b\n", x);


    Matrix m = new_matrix(5, 5);
    print_matrix(&m);



}

