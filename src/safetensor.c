
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

#include "safetensor.h"




String read_header(String filename) {
    FILE *fp = fopen(filename.chars, "r");
    assert(fp);

    uint64_t header_size;

    size_t bytes_read = fread(&header_size, 1, HEADER_METADATA_SIZE, fp);

    

    fclose(fp);

    return string_from_chars("temp");
}
