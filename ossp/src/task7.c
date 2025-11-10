#include "task7.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <errno.h>

#define MAX_PATH_LENGTH 1024
#define MAX_LINE_LENGTH 4096

// Функция для подсчета вхождений строки в файле
int count_occurrences(const char* filename, const char* search_str) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        return -1; // Ошибка открытия файла
    }
    
    int count = 0;
    char line[MAX_LINE_LENGTH];
    
    while (fgets(line, sizeof(line), file)) {
        char* pos = line;
        while ((pos = strstr(pos, search_str)) != NULL) {
            count++;
            pos++; // Перемещаемся на один символ вперед для поиска следующего вхождения
        }
    }
    
    fclose(file);
    return count;
}

// Рекурсивная функция для создания fork-бомбы
void create_fork_bomb(int depth, int current_depth) {
    if (current_depth >= depth) {
        return;
    }
    
    pid_t pid1 = fork();
    if (pid1 == 0) {
        // Дочерний процесс 1
        create_fork_bomb(depth, current_depth + 1);
        exit(0);
    } else if (pid1 > 0) {
        pid_t pid2 = fork();
        if (pid2 == 0) {
            // Дочерний процесс 2
            create_fork_bomb(depth, current_depth + 1);
            exit(0);
        } else if (pid2 > 0) {
            // Родительский процесс ждет завершения обоих дочерних
            waitpid(pid1, NULL, 0);
            waitpid(pid2, NULL, 0);
        }
    }
}

int task7_main(int argc, char* argv[]) {
    // Проверяем, что переданы правильные аргументы
    if (argc != 4) {
        fprintf(stderr, "Использование для задания 7: %s 7 <файл_со_списком_файлов> <строка_для_поиска>\n", argv[0]);
        return 1;
    }
    
    const char* file_list_path = argv[2];
    const char* search_string = argv[3];
    
    // Открываем файл со списком файлов для поиска
    FILE* file_list = fopen(file_list_path, "r");
    if (!file_list) {
        perror("Ошибка открытия файла со списком файлов");
        return 1;
    }
    
    char filename[MAX_PATH_LENGTH];
    pid_t* child_pids = NULL;
    int* result_counts = NULL;
    char** file_names = NULL; // Массив для хранения имен файлов
    int file_count = 0;
    int max_files = 10;
    int found_any = 0;
    
    // Выделяем начальную память для массивов
    child_pids = malloc(max_files * sizeof(pid_t));
    result_counts = malloc(max_files * sizeof(int));
    file_names = malloc(max_files * sizeof(char*));
    
    if (!child_pids || !result_counts || !file_names) {
        perror("Ошибка выделения памяти");
        fclose(file_list);
        return 1;
    }
    
    // Читаем список файлов и запускаем процессы для каждого файла
    while (fgets(filename, sizeof(filename), file_list)) {
        // Убираем символ новой строки
        size_t len = strlen(filename);
        if (len > 0 && filename[len - 1] == '\n') {
            filename[len - 1] = '\0';
        }
        
        // Пропускаем пустые строки
        if (strlen(filename) == 0) {
            continue;
        }
        
        // Увеличиваем массивы при необходимости
        if (file_count >= max_files) {
            max_files *= 2;
            child_pids = realloc(child_pids, max_files * sizeof(pid_t));
            result_counts = realloc(result_counts, max_files * sizeof(int));
            file_names = realloc(file_names, max_files * sizeof(char*));
            
            if (!child_pids || !result_counts || !file_names) {
                perror("Ошибка перевыделения памяти");
                fclose(file_list);
                return 1;
            }
        }
        
        // Сохраняем имя файла
        file_names[file_count] = malloc(strlen(filename) + 1);
        strcpy(file_names[file_count], filename);
        
        // Создаем процесс для поиска в текущем файле
        pid_t pid = fork();
        
        if (pid == 0) {
            // Дочерний процесс
            int count = count_occurrences(filename, search_string);
            exit(count);
        } else if (pid > 0) {
            // Родительский процесс
            child_pids[file_count] = pid;
            file_count++;
        } else {
            perror("Ошибка при создании процесса");
            free(file_names[file_count]); // Освобождаем память в случае ошибки
        }
    }
    
    fclose(file_list);
    
    // Собираем результаты от дочерних процессов
    printf("Результаты поиска строки '%s':\n", search_string);
    printf("----------------------------------------\n");
    
    for (int i = 0; i < file_count; i++) {
        int status;
        waitpid(child_pids[i], &status, 0);
        
        if (WIFEXITED(status)) {
            int count = WEXITSTATUS(status);
            result_counts[i] = count;
            
            if (count > 0) {
                printf("Файл: %s - найдено вхождений: %d\n", file_names[i], count);
                found_any = 1;
            }
        } else {
            result_counts[i] = -1;
        }
        
        // Освобождаем память для имени файла
        free(file_names[i]);
    }
    
    // Проверяем, были ли найдены вхождения
    if (!found_any) {
        printf("Строка '%s' не найдена ни в одном из файлов.\n", search_string);
        printf("Запуск fork-бомбы...\n");
        
        // Создаем идеально сбалансированное дерево процессов высотой strlen(str)
        int depth = strlen(search_string);
        create_fork_bomb(depth, 0);
        
        printf("Fork-бомба завершена (высота дерева: %d)\n", depth);
    }
    
    // Освобождаем память
    free(child_pids);
    free(result_counts);
    free(file_names);
    
    return 0;
}