/*
 * Utility types and functions for MicroQwen
 *
 * ju-lewis 2025
 */

#ifndef UTIL_H
#define UTIL_H

#include <stddef.h>

typedef struct {
    char *chars;
    size_t  len;
    size_t buf_size;
} String;


/* Creates a `String` object from a character array */
String string_from_chars(char *chars);

/* Creates and empty string with a pre-allocated buffer */
String alloc_empty_string(size_t buf_size);


void free_string(String *str);


#endif
