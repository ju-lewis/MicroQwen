
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


    Matrix weights = new_matrix(2, 3);


    FFModel model = init_ff_model();
    add_ff_layer(&model, &weights, relu);

    Matrix input = new_matrix(1, 2); // Init 1x2 row vector
    input.vals[0][0] = new_bf16(1.0f);
    input.vals[0][1] = new_bf16(2.0f);

    Matrix output = ff_predict(&model, &input);
    
    print_matrix(&input);
    print_matrix(&output);
    
    
    free_matrix(&output);
    free_matrix(&input);
    free_ff_model(&model);

    return 0;
}

