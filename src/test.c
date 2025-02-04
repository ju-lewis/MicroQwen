
#include "linalg.h"

#include <stdio.h>


int main() {
    
    bfloat16 x = new_bf16(1.0f);
    bfloat16 y = new_bf16(-2.0f);

    x = add_bf16(x,y);

    printf("%016b\n", x);


    //Matrix m = new_matrix(5, 5);
    //print_matrix(&m);



}

