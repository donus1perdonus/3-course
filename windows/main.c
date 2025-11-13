#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include "task1.h"
#include "task2.h"
#include "task3.h"
#include "task4.h"

void print_usage(const char *program_name) {
    printf("Использование: %s <номер_задания> [аргументы]\n", program_name);
    printf("Доступные задания: 1-4 (Windows версия)\n");
    printf("Примеры:\n");
    printf("  %s 1 -f test.bin                    # Задание 1 с файлом test.bin\n", program_name);
    printf("  %s 2                                # Задание 2\n", program_name);
    printf("  %s 3 source.txt backup.txt          # Задание 3 - копирование файлов\n", program_name);
    printf("  %s 4 data.bin xor8                  # Задание 4 - побайтовая обработка\n", program_name);
}

int main(int argc, char *argv[]) {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    int task_number = atoi(argv[1]);
    
    if (task_number < 1 || task_number > 4) {
        fprintf(stderr, "Ошибка: номер задания должен быть от 1 до 4 (Windows версия)\n");
        print_usage(argv[0]);
        return 1;
    }

    printf("Запуск задания %d (Windows версия)...\n\n", task_number);

    switch (task_number) {
        case 1:
            if (argc < 4 || strcmp(argv[2], "-f") != 0) {
                fprintf(stderr, "Ошибка: ожидается формат '-f <имя_файла>', получено: ");
                for (int i = 2; i < argc; ++i) fprintf(stderr, "%s ", argv[i]);
                fprintf(stderr, "\n");
                printf("Использование: %s 1 -f <имя_файла>\n", argv[0]);
                return 1;
            }
            // Формируем новый argv для task1_main
            char *task1_argv[] = {"task1", "-f", argv[3], NULL};
            task1_main(3, task1_argv);
            break;
        case 2:
            task2_main();
            break;
        case 3:
            if (argc < 4) {
                fprintf(stderr, "Ошибка: для задания 3 требуется указать исходный и целевой файлы\n");
                printf("Использование: %s 3 <исходный_файл> <целевой_файл>\n", argv[0]);
                return 1;
            }
            char *source_file = argv[2];
            char *dest_file = argv[3];
            if (source_file == NULL || dest_file == NULL) {
                fprintf(stderr, "Ошибка: ожидается формат '<исходный_файл> <целевой_файл>', получено: ");
                for (int i = 2; i < argc; ++i) fprintf(stderr, "%s ", argv[i]);
                fprintf(stderr, "\n");
                printf("Использование: %s 3 <исходный_файл> <целевой_файл>\n", argv[0]);
                return 1;
            }
            char absolute_src_path[_MAX_PATH];
            char absolute_dst_path[_MAX_PATH];
            if (_fullpath(absolute_src_path, source_file, _MAX_PATH) != NULL) {
                printf("Абсолютный путь: %s\n", absolute_src_path);
            } else {
                perror("Ошибка преобразования пути");
            }
            if (_fullpath(absolute_dst_path, dest_file, _MAX_PATH) != NULL) {
                printf("Абсолютный путь: %s\n", absolute_dst_path);
            } else {
                perror("Ошибка преобразования пути");
            }
            char *task3_argv[] = {"task3", absolute_src_path, absolute_dst_path, NULL};
            task3_main(3, task3_argv);
            break;
        case 4:
            if (argc < 4) {
                fprintf(stderr, "Ошибка: для задания 4 требуется указать файл и флаг\n");
                printf("Использование: %s 4 <файл> <флаг> [параметры]\n", argv[0]);
                return 1;
            }
            char *file_path = argv[2];
            char *task4_flag = argv[3];
            char *param = argc > 4 ? argv[4] : NULL;
            if (file_path == NULL || task4_flag == NULL) {
                fprintf(stderr, "Ошибка: ожидается формат '<файл> <флаг> [параметры]'");
                printf("Использование: %s 4 <файл> <флаг> [параметры]\n", argv[0]);
                return 1;
            }
            char *task4_argv[5] = {"task4", file_path, task4_flag, param, NULL};
            int task4_argc = (param == NULL) ? 3 : 4;
            task4_main(task4_argc, task4_argv);
            break;
        default:
            fprintf(stderr, "Ошибка: неизвестный номер задания\n");
            return 1;
    }

    return 0;
}
