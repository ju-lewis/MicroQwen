/*
 * safetensor.h defines all types and declares functions related to loading safetensors
 * for MicroQwen
 *
 * ju-lewis 2025
 */
#ifndef SAFETENSOR_H
#define SAFETENSOR_H

#include <stdio.h>

#include "util.h"
#include "linalg.h"
#include "nn.h"


#define HEADER_METADATA_SIZE 8      // `.safetensors` metadata length (first 8 bytes of the file)
#define MAX_TENSOR_NAME_LEN  64     // Longest name a tensor can have in a '.safetensors' file
#define EXPECTED_FIELDS_PER_LINE 5  // The number of fields to be read per line in a 'tensorshape.txt' file
#define TENSORSHAPE_FORMAT_STR "%s [%u, %u] [%lu, %lu]"
#define ALT_TENSORSHAPE_FORMAT_STR "%s [%u] [%lu, %lu]" // This can parse a vector (implicit 1xN size matrix)
#define MAX_LINE_LEN 100

typedef enum {
    INPUT_LAYERNORM,
    ATTN_LAYERNORM,

    MLP_DOWN_PROJ,
    MLP_GATE_PROJ,
    MLP_UP_PROJ,

    K_PROJ,
    K_BIAS,
    Q_PROJ,
    Q_BIAS,
    V_PROJ,
    V_BIAS,
    O_PROJ
} TensorType;



/* Reads a safetensor file header */
String read_header(FILE *fp);

/* Reads the binary data for a bfloat16 matrix from a safetensor file into a `Matrix` object */
Matrix read_binary_matrix(FILE *fp, long offset, unsigned int rows, unsigned int cols);

/* Loads the weights of a decoder model from a safetensor file using the custom representation 
   as an easy-to-parse `map` for finding dimensions and byte offsets (see Python scripts for formatter) */
Decoder load_decoder_from_safetensor(String parsed_format_filename, String safetensor_filename);

/* Parses the name of a tensor to determine its role in a transformer cell */
TensorType get_tensor_type(char *tensor_name);


#endif
