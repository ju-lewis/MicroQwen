/*
 * Utility types and functions for MicroQwen
 *
 * ju-lewis 2025
 */

#ifndef UTIL_H
#define UTIL_H

typedef struct {
    char *chars;
    size_t  len;
    size_t buf_size;
} String;


String string_from_chars(char *chars);

void free_string(String *str);


#endif
