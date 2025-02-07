
#include "linalg.h"
#include "safetensor.h"
#include "util.h"
#include "nn.h"

#include <stdio.h>


int main() {

    //// Load embedding matrix 
    //FILE *fp = fopen("./Qwen2.5-0.5B/model.safetensors", "r");

    //long base_offset = 32288;
    //
    //Matrix m = read_binary_matrix(fp, base_offset, QWEN25_VOCAB_SIZE, 896);
    //fclose(fp);


    //Matrix vec = one_hot_encoding(QWEN25_VOCAB_SIZE, 2159); // 2159 is the token ID for button


    //Matrix embed = naive_matmul(&vec, &m);

    //print_matrix_shape(&embed);
    //print_matrix(&embed);


    //free_matrix(&m);
    //free_matrix(&vec);
    //free_matrix(&embed);
}

