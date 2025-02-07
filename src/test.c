
#include "linalg.h"
#include "safetensor.h"
#include "util.h"

#include <stdio.h>


int main() {

    // Load embedding matrix 
    FILE *fp = fopen("./Qwen2.5-0.5B/model.safetensors", "r");

    long base_offset = 32288;
    
    Matrix m = read_binary_matrix(fp, base_offset, 151936, 896);
    fclose(fp);

    printf("Read matrix\n");

    Matrix vec = new_matrix(1, 151936);

    printf("Created vector\n");

    naive_matmul(&vec, &m);

    printf("Finished matmul\n");


    free_matrix(&m);
    free_matrix(&vec);
}

