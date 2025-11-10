#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>
#include <limits.h>
#include <task8.h>


// Функция для инициализации таблицы файлов
void init_file_table(FileTable *table, int initial_capacity) {
    table->files = malloc(initial_capacity * sizeof(FileInfo));
    if (!table->files) {
        perror("Ошибка выделения памяти");
        exit(1);
    }
    table->count = 0;
    table->capacity = initial_capacity;
}

// Функция для освобождения таблицы файлов
void free_file_table(FileTable *table) {
    if (table->files) {
        free(table->files);
    }
    table->count = 0;
    table->capacity = 0;
}

// Функция для добавления файла в таблицу (если его еще нет)
int add_file_to_table(FileTable *table, const char *name, const char *path, long disk_address) {
    // Проверяем, есть ли уже такой файл в таблице
    for (int i = 0; i < table->count; i++) {
        if (strcmp(table->files[i].path, path) == 0) {
            return 0; // Файл уже существует
        }
    }
    
    // Увеличиваем емкость при необходимости
    if (table->count >= table->capacity) {
        table->capacity *= 2;
        FileInfo *new_files = realloc(table->files, table->capacity * sizeof(FileInfo));
        if (!new_files) {
            perror("Ошибка перевыделения памяти");
            return -1;
        }
        table->files = new_files;
    }
    
    // Заполняем информацию о файле
    FileInfo *file = &table->files[table->count];
    
    // Извлекаем имя и расширение
    const char *last_slash = strrchr(path, '/');
    const char *filename = last_slash ? last_slash + 1 : path;
    
    strncpy(file->name, filename, NAME_MAX);
    file->name[NAME_MAX] = '\0';
    
    // Извлекаем расширение
    const char *dot = strrchr(filename, '.');
    if (dot && dot != filename) { // Есть расширение и это не скрытый файл
        strncpy(file->extension, dot + 1, NAME_MAX);
        file->extension[NAME_MAX] = '\0';
    } else {
        file->extension[0] = '\0'; // Нет расширения
    }
    
    file->disk_address = disk_address;
    strncpy(file->path, path, PATH_MAX);
    file->path[PATH_MAX] = '\0';
    
    table->count++;
    return 1;
}

// Функция для получения дискового адреса файла (inode)
long get_disk_address(const char *path) {
    struct stat st;
    if (stat(path, &st) == 0) {
        return (long)st.st_ino; // Используем inode как дисковый адрес
    }
    return -1;
}

// Рекурсивная функция обхода каталога
void traverse_directory(const char *dir_path, FileTable *table, int recmin, int recmax, int current_depth) {
    // Проверяем глубину рекурсии
    if (current_depth > recmax) {
        return;
    }
    
    DIR *dir = opendir(dir_path);
    if (!dir) {
        fprintf(stderr, "Ошибка открытия каталога %s: %s\n", dir_path, strerror(errno));
        return;
    }
    
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        // Пропускаем текущий и родительский каталоги
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }
        
        // Формируем полный путь
        char full_path[PATH_MAX + 1];
        snprintf(full_path, sizeof(full_path), "%s/%s", dir_path, entry->d_name);
        
        // Получаем информацию о файле
        struct stat st;
        if (stat(full_path, &st) != 0) {
            fprintf(stderr, "Ошибка получения информации о %s: %s\n", full_path, strerror(errno));
            continue;
        }
        
        // Если это файл и мы достигли минимальной глубины, добавляем в таблицу
        if (S_ISREG(st.st_mode) && current_depth >= recmin) {
            long disk_address = get_disk_address(full_path);
            if (disk_address != -1) {
                add_file_to_table(table, entry->d_name, full_path, disk_address);
            }
        }
        
        // Если это каталог и не превышена максимальная глубина, рекурсивно обходим
        if (S_ISDIR(st.st_mode) && current_depth < recmax) {
            traverse_directory(full_path, table, recmin, recmax, current_depth + 1);
        }
    }
    
    closedir(dir);
}

// Функция для вывода информации о файлах
void print_file_table(const FileTable *table) {
    if (table->count == 0) {
        printf("Файлы не найдены.\n");
        return;
    }
    
    // Заголовок таблицы
    printf("%-30s %-15s %-15s %s\n", "Имя файла", "Расширение", "Дисковый адрес", "Полный путь");
    printf("%-30s %-15s %-15s %s\n", "---------", "----------", "--------------", "-----------");
    
    // Данные файлов
    for (int i = 0; i < table->count; i++) {
        const FileInfo *file = &table->files[i];
        printf("%-30s %-15s %-15ld %s\n", 
               file->name, 
               file->extension[0] ? file->extension : "(нет)", 
               file->disk_address,
               file->path);
    }
    
    printf("\nВсего файлов: %d\n", table->count);
}

// Функция сравнения для сортировки файлов по имени
int compare_files(const void *a, const void *b) {
    const FileInfo *fileA = (const FileInfo *)a;
    const FileInfo *fileB = (const FileInfo *)b;
    return strcmp(fileA->name, fileB->name);
}

// Основная функция
int task8_main(int argc, char *argv[]) {
    // Проверка аргументов командной строки
    if (argc < 5) {
        fprintf(stderr, "Использование: %s <recmin> <recmax> <путь1> [путь2 ...]\n", argv[0]);
        fprintf(stderr, "  recmin  - минимальная глубина рекурсии (>= 0)\n");
        fprintf(stderr, "  recmax  - максимальная глубина рекурсии (>= recmin)\n");
        fprintf(stderr, "  путьN   - один или несколько путей для обхода\n");
        return 1;
    }
    
    // Парсинг recmin и recmax
    int recmin = atoi(argv[2]);
    int recmax = atoi(argv[3]);
    
    if (recmin < 0 || recmax < recmin) {
        fprintf(stderr, "Ошибка: некорректные значения recmin и recmax\n");
        fprintf(stderr, "Должно быть: 0 <= recmin <= recmax\n");
        return 1;
    }
    
    // Инициализация таблицы файлов
    FileTable table;
    init_file_table(&table, 100);
    
    // Обход каждого указанного каталога
    for (int i = 4; i < argc; i++) {
        char *dir_path = argv[i];
        
        // Проверяем, что путь существует и это каталог
        struct stat st;
        if (stat(dir_path, &st) != 0) {
            fprintf(stderr, "Ошибка: путь '%s' не существует\n", dir_path);
            continue;
        }
        
        if (!S_ISDIR(st.st_mode)) {
            fprintf(stderr, "Ошибка: '%s' не является каталогом\n", dir_path);
            continue;
        }
        
        printf("Обход каталога: %s (глубина: %d-%d)\n", dir_path, recmin, recmax);
        printf("========================================\n");
        
        traverse_directory(dir_path, &table, recmin, recmax, 0);
    }
    
    // Сортируем файлы по имени для удобного просмотра
    if (table.count > 0) {
        qsort(table.files, table.count, sizeof(FileInfo), compare_files);
    }
    
    // Выводим результаты
    print_file_table(&table);
    
    // Освобождаем память
    free_file_table(&table);
    
    return 0;
}