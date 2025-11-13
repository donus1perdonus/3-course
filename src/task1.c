#include "task1.h"
#include <stdio.h>
#include <string.h>

void print_file_position(FILE *file) {
    long position = ftell(file);
    if (position == -1L) {
        perror("Ошибка при получении позиции в файле");
        return;
    }
    printf("Текущая позиция в файле: %ld\n", position);
}

void task1_main(int argc, char *argv[]) {
    printf("=== Задание 1 ===\n");
    
    if (argc < 3) {
        fprintf(stderr, "Использование: task1 -f <имя_файла>\n");
        return;
    }

    if (strcmp(argv[1], "-f") != 0) {
        fprintf(stderr, "Ошибка: ожидался флаг -f, получено '%s'\n", argv[1]);
        fprintf(stderr, "Использование: task1 -f <имя_файла>\n");
        return;
    }

    const char *filename = argv[2];
    unsigned char data[] = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5};
    size_t data_size = sizeof(data);

    // Создание и запись файла
    FILE *file = fopen(filename, "wb");
    if (file == NULL) {
        perror("Ошибка при создании файла");
        return;
    }

    size_t written = fwrite(data, 1, data_size, file);
    if (written != data_size) {
        perror("Ошибка при записи в файл");
        fclose(file);
        return;
    }

    if (fclose(file) != 0) {
        perror("Ошибка при закрытии файла");
        return;
    }

    printf("Файл успешно создан с данными: ");
    for (size_t i = 0; i < data_size; i++) {
        printf("%d ", data[i]);
    }
    printf("\n\n");

    // Чтение файла с побайтовым выводом
    file = fopen(filename, "rb");
    if (file == NULL) {
        perror("Ошибка при открытии файла для чтения");
        return;
    }

    printf("Чтение файла по байтам:\n");
    unsigned char byte;
    size_t position = 0;
    
    while (fread(&byte, 1, 1, file) == 1) {
        printf("Позиция: %zu, Байт: %d\n", position, byte);
        print_file_position(file);
        printf("Индикатор конца файла (EOF): %d\n", feof(file));
        printf("Индикатор ошибки: %d\n", ferror(file));
        printf("\n");
        position++;
    }

    if (ferror(file)) {
        perror("Ошибка при чтении файла");
        fclose(file);
        return;
    }

    if (fclose(file) != 0) {
        perror("Ошибка при закрытии файла");
        return;
    }

    // Повторное открытие и использование fseek
    file = fopen(filename, "rb");
    if (file == NULL) {
        perror("Ошибка при повторном открытии файла");
        return;
    }

    // Перемещение на 3 байта от начала
    if (fseek(file, 3, SEEK_SET) != 0) {
        perror("Ошибка при перемещении по файлу (fseek)");
        fclose(file);
        return;
    }

    printf("После перехода (fseek) к позиции 3:\n");
    print_file_position(file);
    printf("\n");

    // Чтение 4 байтов
    unsigned char buffer[4];
    size_t bytes_read = fread(buffer, 1, 4, file);
    
    if (bytes_read != 4) {
        if (ferror(file)) {
            perror("Ошибка при чтении из файла");
        } else {
            printf("Удалось прочитать только %zu байта из 4\n", bytes_read);
        }
        fclose(file);
        return;
    }

    printf("После чтения 4 байтов:\n");
    print_file_position(file);
    printf("Буфер содержит: ");
    for (size_t i = 0; i < 4; i++) {
        printf("%d ", buffer[i]);
    }
    printf("\n");

    if (fclose(file) != 0) {
        perror("Ошибка при закрытии файла");
        return;
    }
}