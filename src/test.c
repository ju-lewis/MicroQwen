
#include <stdio.h>

#include "linalg.h"
#include "safetensor.h"
#include "util.h"
#include "nn.h"



int main() {

    //TODO: Implement functions required to make this work
    Decoder model = load_decoder_from_safetensor(parsed_format_map_filename, 
                                             safetensor_filename);
    Matrix sequence = load_tokenised_sequence(token_filename);

    // Load embedding matrix from .safetensors
    FILE *fp = fopen("./Qwen2.5-0.5B/model.safetensors", "r");
    long embedding_offset = 32288;
    Matrix embedding_matrix = read_binary_matrix(fp, embedding_offset,
                                                 QWEN25_VOCAB_SIZE, 896);
    fclose(fp);
    
    Matrix sequence_embeddings = naive_matmul(&sequence, &embedding_matrix);

    Matrix prev_sequence = sequence_embeddings;
    Matrix next = empty_matrix(); // Creates matrix and sets `vals` to NULL

    // Now we can start running token generation (ABSTRACT THIS TO A GENERATION FN)
    while(!is_eos_token(next) || limit_reached) {
        next = predict_next_embedding(&model, &sequence_embeddings);

        // Append `next` values to the sequence
        free_matrix(&prev_sequence);
        prev_sequence = sequence_embeddings;
        sequence_embeddings = concat_matrices(/* .......... */);

        // Free `next`
        free_matrix(&next);
    }

    // Re
    
    return 0;
}

