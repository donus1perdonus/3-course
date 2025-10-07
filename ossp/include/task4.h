#ifndef TASK4_H
#define TASK4_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stdint.h>

// Функции для побайтовой обработки файлов
uint8_t xor8(const char* file_path);
uint32_t xorodd(const char* file_path);
uint32_t mask_count(const char* file_path, uint32_t hex_mask);
int is_prime(uint8_t n);
void print_task4_usage(const char* program_name);
int validate_file_exists_task4(const char* file_path);
uint32_t parse_hex_mask(const char* hex_str);

void task4_main(int argc, char *argv[]);

#endif // TASK4_H
