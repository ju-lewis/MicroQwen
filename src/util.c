
#include "util.h"
#include <string.h>
#include <stdlib.h>
#include <assert.h>


String string_from_chars(char *chars) {

    size_t num_chars = strlen(chars);

    String str;
    str.chars = (char *)malloc(sizeof(char) * (num_chars + 1)); // Add an extra character for null

    assert(str.chars);

    strlcpy(str.chars, chars, num_chars);
    str.buf_size = num_chars + 1;
    str.len = num_chars;
    
    return str;
}


void free_string(String *str) {
    free(str->chars);
    str->chars = NULL;
    str->buf_size = 0;
    str->len = 0;
}



