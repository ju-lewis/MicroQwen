
#include <stdio.h>

#include "linalg.h"
#include "safetensor.h"
#include "util.h"
#include "nn.h"



int main() {

    //// Load embedding matrix 
    //FILE *fp = fopen("./Qwen2.5-0.5B/model.safetensors", "r");

    //long base_offset = 32288;
    //
    //Matrix m = read_binary_matrix(fp, base_offset, QWEN25_VOCAB_SIZE, 896);
    //fclose(fp);

    Matrix a = new_matrix(2, 2);
    Matrix b = new_matrix(2, 2);

    Matrix ms[2] = {a, b};

    Matrix c = concat_matrices(ms, 2, COLUMN);

    print_matrix(&c);

    return 0;
}

