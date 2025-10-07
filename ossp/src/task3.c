#include "task3.h"

// Вывод справки по использованию
void print_task3_usage(const char* program_name) {
    printf("Использование: %s <исходный_файл> <целевой_файл>\n", program_name);
    printf("  исходный_файл  - файл, который необходимо скопировать\n");
    printf("  целевой_файл   - файл, в который будет выполнено копирование\n");
    printf("\nПримеры:\n");
    printf("  %s source.txt backup.txt\n", program_name);
    printf("  %s data.bin copy.bin\n", program_name);
}

// Проверка существования файла
int validate_file_exists(const char* file_path) {
    FILE* file = fopen(file_path, "rb");
    if (file == NULL) {
        return 0;
    }
    fclose(file);
    return 1;
}

// Получение размера файла
long get_file_size(const char* file_path) {
    FILE* file = fopen(file_path, "rb");
    if (file == NULL) {
        return -1;
    }
    
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fclose(file);
    return size;
}

// Копирование файла
int copy_file(const char* source_path, const char* dest_path) {
    FILE* source_file = NULL;
    FILE* dest_file = NULL;
    char buffer[BUFFER_SIZE];
    size_t bytes_read, bytes_written;
    int result = 1; // 1 = успех, 0 = ошибка
    
    printf("Копирование файла: %s -> %s\n", source_path, dest_path);
    
    // Открываем исходный файл
    source_file = fopen(source_path, "rb");
    if (source_file == NULL) {
        fprintf(stderr, "Ошибка: не удалось открыть исходный файл '%s': %s\n", 
                source_path, strerror(errno));
        return 0;
    }
    
    // Открываем целевой файл
    dest_file = fopen(dest_path, "wb");
    if (dest_file == NULL) {
        fprintf(stderr, "Ошибка: не удалось создать целевой файл '%s': %s\n", 
                dest_path, strerror(errno));
        fclose(source_file);
        return 0;
    }
    
    // Копируем данные блоками
    while ((bytes_read = fread(buffer, 1, BUFFER_SIZE, source_file)) > 0) {
        bytes_written = fwrite(buffer, 1, bytes_read, dest_file);
        if (bytes_written != bytes_read) {
            fprintf(stderr, "Ошибка: не удалось записать данные в целевой файл: %s\n", 
                    strerror(errno));
            result = 0;
            break;
        }
    }
    
    // Проверяем ошибки чтения
    if (ferror(source_file)) {
        fprintf(stderr, "Ошибка при чтении исходного файла: %s\n", strerror(errno));
        result = 0;
    }
    
    // Проверяем ошибки записи
    if (ferror(dest_file)) {
        fprintf(stderr, "Ошибка при записи в целевой файл: %s\n", strerror(errno));
        result = 0;
    }
    
    // Закрываем файлы
    if (fclose(source_file) != 0) {
        fprintf(stderr, "Ошибка при закрытии исходного файла: %s\n", strerror(errno));
        result = 0;
    }
    
    if (fclose(dest_file) != 0) {
        fprintf(stderr, "Ошибка при закрытии целевого файла: %s\n", strerror(errno));
        result = 0;
    }
    
    if (result) {
        printf("Файл успешно скопирован.\n");
    }
    
    return result;
}

// Сравнение файлов для проверки идентичности
int compare_files(const char* file1_path, const char* file2_path) {
    FILE* file1 = NULL;
    FILE* file2 = NULL;
    char buffer1[BUFFER_SIZE];
    char buffer2[BUFFER_SIZE];
    size_t bytes_read1, bytes_read2;
    int result = 1; // 1 = идентичны, 0 = различны
    
    printf("Проверка идентичности файлов...\n");
    
    // Открываем первый файл
    file1 = fopen(file1_path, "rb");
    if (file1 == NULL) {
        fprintf(stderr, "Ошибка: не удалось открыть файл '%s': %s\n", 
                file1_path, strerror(errno));
        return 0;
    }
    
    // Открываем второй файл
    file2 = fopen(file2_path, "rb");
    if (file2 == NULL) {
        fprintf(stderr, "Ошибка: не удалось открыть файл '%s': %s\n", 
                file2_path, strerror(errno));
        fclose(file1);
        return 0;
    }
    
    // Сравниваем файлы блоками
    while (1) {
        bytes_read1 = fread(buffer1, 1, BUFFER_SIZE, file1);
        bytes_read2 = fread(buffer2, 1, BUFFER_SIZE, file2);
        
        // Если количество прочитанных байт разное, файлы различны
        if (bytes_read1 != bytes_read2) {
            result = 0;
            break;
        }
        
        // Если достигли конца файла
        if (bytes_read1 == 0) {
            break;
        }
        
        // Сравниваем содержимое буферов
        if (memcmp(buffer1, buffer2, bytes_read1) != 0) {
            result = 0;
            break;
        }
    }
    
    // Проверяем ошибки чтения
    if (ferror(file1) || ferror(file2)) {
        fprintf(stderr, "Ошибка при чтении файлов для сравнения: %s\n", strerror(errno));
        result = 0;
    }
    
    // Закрываем файлы
    fclose(file1);
    fclose(file2);
    
    if (result) {
        printf("✓ Файлы идентичны.\n");
    } else {
        printf("✗ Файлы различны.\n");
    }
    
    return result;
}

void task3_main(int argc, char *argv[]) {
    printf("=== Задание 3: Копирование файлов ===\n");
    
    // Проверяем количество аргументов
    if (argc < 3) {
        fprintf(stderr, "Ошибка: недостаточно аргументов.\n\n");
        print_task3_usage("task3");
        return;
    }
    
    if (argc > 3) {
        fprintf(stderr, "Ошибка: слишком много аргументов.\n\n");
        print_task3_usage("task3");
        return;
    }
    
    const char* source_path = argv[1];
    const char* dest_path = argv[2];
    
    // Проверяем, что исходный файл существует
    if (!validate_file_exists(source_path)) {
        fprintf(stderr, "Ошибка: исходный файл '%s' не существует или недоступен.\n", source_path);
        return;
    }
    
    // Получаем размер исходного файла
    long source_size = get_file_size(source_path);
    if (source_size < 0) {
        fprintf(stderr, "Ошибка: не удалось определить размер исходного файла.\n");
        return;
    }
    
    printf("Размер исходного файла: %ld байт\n", source_size);
    
    // Проверяем, что целевой файл не совпадает с исходным
    if (strcmp(source_path, dest_path) == 0) {
        fprintf(stderr, "Ошибка: исходный и целевой файлы не могут быть одинаковыми.\n");
        return;
    }
    
    // Выполняем копирование
    if (!copy_file(source_path, dest_path)) {
        fprintf(stderr, "Копирование не удалось.\n");
        return;
    }
    
    // Проверяем размер целевого файла
    long dest_size = get_file_size(dest_path);
    if (dest_size < 0) {
        fprintf(stderr, "Ошибка: не удалось определить размер целевого файла.\n");
        return;
    }
    
    printf("Размер целевого файла: %ld байт\n", dest_size);
    
    // Проверяем, что размеры файлов совпадают
    if (source_size != dest_size) {
        fprintf(stderr, "Ошибка: размеры файлов не совпадают (%ld != %ld).\n", 
                source_size, dest_size);
        return;
    }
    
    // Проверяем идентичность файлов
    if (!compare_files(source_path, dest_path)) {
        fprintf(stderr, "Ошибка: файлы не идентичны после копирования.\n");
        return;
    }
    
    printf("\n✓ Копирование выполнено успешно!\n");
    printf("✓ Исходный и целевой файлы идентичны.\n");
}
