
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

#include "safetensor.h"
#include "linalg.h"



/* Reads a safetensor file header */
String read_header(FILE *fp) {
    assert(fp);

    uint64_t header_size;

    // Read the size of the file header
    size_t bytes_read = fread(&header_size, 1, HEADER_METADATA_SIZE, fp);
    assert(bytes_read == HEADER_METADATA_SIZE);

    // Read the JSON UTF-8 header into a String (all chars are in ASCII range)
    String header = alloc_empty_string(header_size + 1);
    fread(header.chars, sizeof(char), header_size, fp);

    header.len = header_size;
    
    return header;
}

/* Reads the binary data for a bfloat16 matrix from a safetensor file into a `Matrix` object */
Matrix read_binary_matrix(FILE *fp, long offset, unsigned int rows, unsigned int cols) {
    
    Matrix m = new_matrix(rows, cols);
    
    fseek(fp, offset, SEEK_SET);

    for (unsigned int i=0; i < rows; i++) {
        fread(m.vals[i], sizeof(bfloat16), cols, fp);
    }


    return m;
}



/* Loads the weights of a decoder model from a safetensor file using the custom representation 
   as an easy-to-parse `map` for finding dimensions and byte offsets (see Python scripts for formatter) */
Decoder load_decoder_from_safetensor(String parsed_format_filename, String safetensor_filename) {

    FILE *map_fp = fopen(parsed_format_filename.chars, "r");
    FILE *tensor_fp = fopen(safetensor_filename.chars, "r");
    assert(map_fp);
    assert(tensor_fp);

    Decoder d;

    // Get the data offset by reading the safetensor file header length
    // (All)
    uint64_t data_offset;
    fread(&data_offset, sizeof(uint64_t), 1, tensor_fp);
    
    char line[MAX_LINE_LEN] = {0};

    // Define tensor metadata
    size_t mat_start, mat_end;
    unsigned int rows, cols;
    char tensor_name[MAX_TENSOR_NAME_LEN];
    int fields_read, 
        lines_parsed = 0;


    // Read embedding matrix metadata
    assert(fgets(line, MAX_LINE_LEN, map_fp));

    fields_read = sscanf(line, TENSORSHAPE_FORMAT_STR, tensor_name, &rows, &cols, &mat_start, &mat_end);
    if (fields_read != EXPECTED_FIELDS_PER_LINE) {
        fprintf(stderr, "Failed to parse %s at line %d. Only parsed %d fields out of %d expected.\n", 
                parsed_format_filename.chars, lines_parsed+1, fields_read, EXPECTED_FIELDS_PER_LINE);
    }
    lines_parsed++;

    // Load the embedding matrix
    d.embedding_matrix = read_binary_matrix(tensor_fp, (long)(data_offset + mat_start), rows, cols);
    

    int line_describes_vector = 0;
    
    while (fgets(line, MAX_LINE_LEN, map_fp)) {

        if (sscanf(line, TENSORSHAPE_FORMAT_STR, tensor_name, &rows, &cols, &mat_start, &mat_end) == EXPECTED_FIELDS_PER_LINE) {
            // First attempt to parse a rank 2 tensor (n x m matrix)
            line_describes_vector = 0;
        } else if (sscanf(line, ALT_TENSORSHAPE_FORMAT_STR, tensor_name, &cols, &mat_start, &mat_end) == EXPECTED_FIELDS_PER_LINE - 1) {
            // If that fails, attempt to parse a rank 1 tensor (1 x n vector)
            line_describes_vector = 1;
        } else {
            // Otherwise we failed parsing the line entirely
            fprintf(stderr, "Failed to parse %s at line %d \"%s\".\n", parsed_format_filename.chars, lines_parsed + 1, line);
        }



        // Determine which layer we're currently parsing
        int layer_num = 0;
        sscanf(tensor_name, "%*s.%*s%d", &layer_num);
        //printf("Currently parsing layer: %d\n", layer_num);



        lines_parsed++;
    }
    printf("Done parsing.\n");
    


    fclose(map_fp);
    fclose(tensor_fp);

    return d;
}



