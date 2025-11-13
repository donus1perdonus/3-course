#ifndef TASK8_H
#define TASK8_H

// Определяем наши константы, если они не определены системой
#ifndef NAME_MAX
#define NAME_MAX 255
#endif

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

// Структура для хранения информации о файле
typedef struct {
    char name[NAME_MAX + 1];
    char extension[NAME_MAX + 1];
    long disk_address;
    char path[PATH_MAX + 1];
} FileInfo;

// Структура для хранения уникальных файлов
typedef struct {
    FileInfo *files;
    int count;
    int capacity;
} FileTable;

void init_file_table(FileTable *table, int initial_capacity);

void free_file_table(FileTable *table);

int add_file_to_table(FileTable *table, const char *path, long disk_address);

long get_disk_address(const char *path);

void traverse_directory(const char *dir_path, FileTable *table, int recmin, int recmax, int current_depth);

void print_file_table(const FileTable *table);

int compare_files(const void *a, const void *b);

int task8_main(int argc, char *argv[]);

#endif // TASK8_H
