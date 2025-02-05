
#include "linalg.h"
#include "safetensor.h"
#include "util.h"

#include <stdio.h>


int main() {

    String s = string_from_chars("./Qwen2.5-0.5B/model.safetensors");
    
    String header = read_header(s);

    
    printf(header.chars);


    free_string(&s);
    free_string(&header);
}

