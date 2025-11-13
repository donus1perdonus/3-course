#ifndef TASK3_H
#define TASK3_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#define BUFFER_SIZE 4096

// Функции для копирования файлов
int copy_file(const char* source_path, const char* dest_path);
int compare_files(const char* file1_path, const char* file2_path);
void print_task3_usage(const char* program_name);
int validate_file_exists(const char* file_path);
long get_file_size(const char* file_path);

void task3_main(int argc, char *argv[]);

#endif // TASK3_H
