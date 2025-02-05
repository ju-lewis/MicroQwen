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


#define HEADER_METADATA_SIZE 8


String read_header(String filename);

Matrix read_binary_matrix(FILE *fp, long offset, unsigned int rows, unsigned int cols);


#endif
