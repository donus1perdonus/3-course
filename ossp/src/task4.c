#include "task4.h"

// Вывод справки по использованию
void print_task4_usage(const char* program_name) {
    printf("Использование: %s <файл> <флаг> [параметры]\n", program_name);
    printf("  файл          - путь к входному файлу\n");
    printf("  флаг          - действие для выполнения:\n");
    printf("    xor8        - сложение по модулю 2 всех байтов файла\n");
    printf("    xorodd      - сложение по модулю 2 четырехбайтовых подпоследовательностей\n");
    printf("                  с простыми числами\n");
    printf("    mask <hex>  - подсчет четырехбайтовых чисел, соответствующих маске\n");
    printf("\nПримеры:\n");
    printf("  %s data.bin xor8\n", program_name);
    printf("  %s data.bin xorodd\n", program_name);
    printf("  %s data.bin mask FF00FF00\n", program_name);
}

// Проверка существования файла
int validate_file_exists_task4(const char* file_path) {
    FILE* file = fopen(file_path, "rb");
    if (file == NULL) {
        return 0;
    }
    fclose(file);
    return 1;
}

// Проверка, является ли число простым
int is_prime(uint8_t n) {
    if (n < 2) return 0;
    if (n == 2) return 1;
    if (n % 2 == 0) return 0;
    
    for (int i = 3; i * i <= n; i += 2) {
        if (n % i == 0) return 0;
    }
    return 1;
}

// Парсинг шестнадцатеричной маски
uint32_t parse_hex_mask(const char* hex_str) {
    return (uint32_t)strtoul(hex_str, NULL, 16);
}

// XOR8 - сложение по модулю 2 всех байтов файла
uint8_t xor8(const char* file_path) {
    FILE* file = fopen(file_path, "rb");
    if (file == NULL) {
        fprintf(stderr, "Ошибка: не удалось открыть файл '%s': %s\n", 
                file_path, strerror(errno));
        return 0;
    }
    
    uint8_t result = 0;
    uint8_t byte;
    
    while (fread(&byte, 1, 1, file) == 1) {
        result ^= byte;
    }
    
    if (ferror(file)) {
        fprintf(stderr, "Ошибка при чтении файла: %s\n", strerror(errno));
        fclose(file);
        return 0;
    }
    
    fclose(file);
    return result;
}

// XORODD - сложение по модулю 2 четырехбайтовых подпоследовательностей с простыми числами
uint32_t xorodd(const char* file_path) {
    FILE* file = fopen(file_path, "rb");
    if (file == NULL) {
        fprintf(stderr, "Ошибка: не удалось открыть файл '%s': %s\n", 
                file_path, strerror(errno));
        return 0;
    }
    
    uint32_t result = 0;
    uint8_t buffer[4];
    size_t bytes_read;
    
    // Читаем файл по 4 байта
    while ((bytes_read = fread(buffer, 1, 4, file)) == 4) {
        // Проверяем, есть ли среди байтов простое число
        int has_prime = 0;
        for (int i = 0; i < 4; i++) {
            if (is_prime(buffer[i])) {
                has_prime = 1;
                break;
            }
        }
        
        // Если есть простое число, выполняем XOR с четырехбайтовым числом
        if (has_prime) {
            uint32_t four_byte_value = 0;
            for (int i = 0; i < 4; i++) {
                four_byte_value |= ((uint32_t)buffer[i]) << (i * 8);
            }
            result ^= four_byte_value;
        }
    }
    
    // Обрабатываем оставшиеся байты, если файл не кратен 4
    if (bytes_read > 0 && bytes_read < 4) {
        // Дополняем нулями до 4 байт
        for (size_t i = bytes_read; i < 4; i++) {
            buffer[i] = 0;
        }
        
        int has_prime = 0;
        for (int i = 0; i < 4; i++) {
            if (is_prime(buffer[i])) {
                has_prime = 1;
                break;
            }
        }
        
        if (has_prime) {
            uint32_t four_byte_value = 0;
            for (int i = 0; i < 4; i++) {
                four_byte_value |= ((uint32_t)buffer[i]) << (i * 8);
            }
            result ^= four_byte_value;
        }
    }
    
    if (ferror(file)) {
        fprintf(stderr, "Ошибка при чтении файла: %s\n", strerror(errno));
        fclose(file);
        return 0;
    }
    
    fclose(file);
    return result;
}

// MASK - подсчет четырехбайтовых чисел, соответствующих маске
uint32_t mask_count(const char* file_path, uint32_t hex_mask) {
    FILE* file = fopen(file_path, "rb");
    if (file == NULL) {
        fprintf(stderr, "Ошибка: не удалось открыть файл '%s': %s\n", 
                file_path, strerror(errno));
        return 0;
    }
    
    uint32_t count = 0;
    uint32_t four_byte_value;
    
    // Читаем файл по 4 байта
    while (fread(&four_byte_value, 4, 1, file) == 1) {
        // Проверяем соответствие маске: (value & mask) == mask
        if ((four_byte_value & hex_mask) == hex_mask) {
            count++;
        }
    }
    
    if (ferror(file)) {
        fprintf(stderr, "Ошибка при чтении файла: %s\n", strerror(errno));
        fclose(file);
        return 0;
    }
    
    fclose(file);
    return count;
}

void task4_main(int argc, char *argv[]) {
    printf("=== Задание 4: Побайтовая обработка файлов ===\n");
    
    // Проверяем количество аргументов
    if (argc < 3) {
        fprintf(stderr, "Ошибка: недостаточно аргументов.\n\n");
        print_task4_usage("task4");
        return;
    }
    
    const char* file_path = argv[1];
    const char* flag = argv[2];
    
    // Проверяем, что файл существует
    if (!validate_file_exists_task4(file_path)) {
        fprintf(stderr, "Ошибка: файл '%s' не существует или недоступен.\n", file_path);
        return;
    }
    
    printf("Обработка файла: %s\n", file_path);
    printf("Флаг: %s\n", flag);
    
    if (strcmp(flag, "xor8") == 0) {
        if (argc != 3) {
            fprintf(stderr, "Ошибка: для флага 'xor8' не требуются дополнительные параметры.\n");
            return;
        }
        
        uint8_t result = xor8(file_path);
        printf("Результат XOR8: 0x%02X (%d)\n", result, result);
        
    } else if (strcmp(flag, "xorodd") == 0) {
        if (argc != 3) {
            fprintf(stderr, "Ошибка: для флага 'xorodd' не требуются дополнительные параметры.\n");
            return;
        }
        
        uint32_t result = xorodd(file_path);
        printf("Результат XORODD: 0x%08X (%u)\n", result, result);
        
    } else if (strcmp(flag, "mask") == 0) {
        if (argc != 4) {
            fprintf(stderr, "Ошибка: для флага 'mask' требуется указать шестнадцатеричную маску.\n");
            printf("Использование: %s <файл> mask <hex>\n", "task4");
            return;
        }
        
        const char* hex_str = argv[3];
        uint32_t mask = parse_hex_mask(hex_str);
        
        // Проверяем корректность парсинга
        if (mask == 0 && strcmp(hex_str, "0") != 0) {
            fprintf(stderr, "Ошибка: неверный формат шестнадцатеричной маски '%s'.\n", hex_str);
            return;
        }
        
        uint32_t count = mask_count(file_path, mask);
        printf("Маска: 0x%08X\n", mask);
        printf("Количество совпадений: %u\n", count);
        
    } else {
        fprintf(stderr, "Ошибка: неизвестный флаг '%s'.\n", flag);
        printf("Доступные флаги: xor8, xorodd, mask\n");
        print_task4_usage("task4");
        return;
    }
    
    printf("\n✓ Обработка завершена успешно!\n");
}
