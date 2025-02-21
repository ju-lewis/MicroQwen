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


#define HEADER_METADATA_SIZE 8

/* Reads a safetensor file header */
String read_header(FILE *fp);

/* Reads the binary data for a bfloat16 matrix from a safetensor file into a `Matrix` object */
Matrix read_binary_matrix(FILE *fp, long offset, unsigned int rows, unsigned int cols);

/* Loads the weights of a decoder model from a safetensor file using the custom representation 
   as an easy-to-parse `map` for finding dimensions and byte offsets (see Python scripts for formatter) */
Decoder load_decoder_from_safetensor(String parsed_format_filename, String safetensor_filename);


#endif
