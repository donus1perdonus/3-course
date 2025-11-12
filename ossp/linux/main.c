#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "task5.h"
#include "task6.h"
#include "task7.h"
#include "task8.h"
#include "task9.h"
#include "task10.h"
#include "task11.h"
#include "task12.h"

void print_usage(const char *program_name) {
    printf("Использование: %s <номер_задания> [аргументы]\n", program_name);
    printf("Доступные задания: 5-8 (Linux версия)\n");
    printf("Примеры:\n");
    printf("  %s 5                                # Задание 5\n", program_name);
    printf("  %s 6                                # Задание 6\n", program_name);
    printf("  %s 7                                # Задание 7\n", program_name);
    printf("  %s 8                                # Задание 8\n", program_name);
    printf("  %s 9                                # Задание 9\n", program_name);
    printf("  %s 10                               # Задание 10\n", program_name);
    printf("  %s 11                               # Задание 11\n", program_name);
    printf("  %s 12                               # Задание 12\n", program_name);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    int task_number = atoi(argv[1]);
    
    if (task_number < 5 || task_number > 8) {
        fprintf(stderr, "Ошибка: номер задания должен быть от 5 до 8 (Linux версия)\n");
        print_usage(argv[0]);
        return 1;
    }

    printf("Запуск задания %d (Linux версия)...\n\n", task_number);

    switch (task_number) {
        case 5:
            task5_main();
            break;
        case 6:
            task6_main();
            break;
        case 7:
            task7_main(argc, argv);
            break;
        case 8:
            task8_main(argc, argv);
            break;
        case 9:
            task9_main(argc, argv);
            break;
        case 10:
            task10_main(argc, argv);
            break;
        case 11:
            task11_main(argc, argv);
            break;
        case 12:
            task12_main(argc, argv);
            break;
        default:
            fprintf(stderr, "Ошибка: неизвестный номер задания\n");
            return 1;
    }

    return 0;
}
