
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

    // Test padding a large row vector
    power_2_pad_matrix(&m);



    free_matrix(&m);
}

