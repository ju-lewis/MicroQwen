
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

#include "safetensor.h"




String read_header(String filename) {
    FILE *fp = fopen(filename.chars, "r");
    assert(fp);

    uint64_t header_size;

    // Read the size of the file header
    size_t bytes_read = fread(&header_size, 1, HEADER_METADATA_SIZE, fp);
    assert(bytes_read == HEADER_METADATA_SIZE);

    // Read the JSON UTF-8 header into a String (all chars are in ASCII range)
    String header = alloc_empty_string(header_size + 1);
    fread(header.chars, sizeof(char), header_size, fp);
    
    fclose(fp);

    return header;
}
