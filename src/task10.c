#include "task10.h"

// Функция серверного процесса
static int server_process(int read_fd) {
    char buffer[1024];
    char line_buffer[1024] = {0};  // Буфер для неполных строк
    size_t line_buffer_len = 0;
    int length_counts[101] = {0}; // Индекс = длина строки, значение = количество
    ssize_t bytes_read;
    int found_length = -1;
    
    printf("[Сервер] Начинаю чтение из канала...\n");
    
    // Читаем строки из канала
    while (found_length == -1) {
        bytes_read = read(read_fd, buffer, sizeof(buffer) - 1);
        
        if (bytes_read == -1) {
            perror("[Сервер] Ошибка чтения из канала");
            close(read_fd);
            return 1;
        }
        
        if (bytes_read == 0) {
            // Канал закрыт клиентом
            printf("[Сервер] Канал закрыт клиентом\n");
            break;
        }
        
        // Добавляем завершающий нулевой символ
        buffer[bytes_read] = '\0';
        
        // Объединяем с предыдущим остатком, если есть
        char *data_start = buffer;
        if (line_buffer_len > 0) {
            // Копируем остаток в начало буфера для обработки
            memmove(buffer, line_buffer, line_buffer_len);
            data_start = buffer;
            bytes_read += line_buffer_len;
            buffer[bytes_read] = '\0';
            line_buffer_len = 0;
        }
        
        // Обрабатываем все полные строки (заканчивающиеся '\n')
        char *line = data_start;
        char *newline;
        
        while ((newline = strchr(line, '\n')) != NULL) {
            *newline = '\0';
            size_t line_length = strlen(line);
            
            if (line_length > 0 && line_length <= MAX_LENGTH) {
                length_counts[line_length]++;
                printf("[Сервер] Получена строка длины %zu (всего таких строк: %d)\n", 
                       line_length, length_counts[line_length]);
                
                // Проверяем, достигли ли мы требуемого количества
                if (length_counts[line_length] >= REQUIRED_COUNT) {
                    found_length = (int)line_length;
                    printf("[Сервер] Получено %d строк длины %d. Завершаю работу.\n", 
                           REQUIRED_COUNT, found_length);
                    break;
                }
            }
            
            line = newline + 1;
        }
        
        // Сохраняем остаток (неполную строку) для следующей итерации
        if (found_length == -1 && line < data_start + bytes_read) {
            size_t remainder_len = (data_start + bytes_read) - line;
            if (remainder_len > 0 && remainder_len < sizeof(line_buffer)) {
                memcpy(line_buffer, line, remainder_len);
                line_buffer[remainder_len] = '\0';
                line_buffer_len = remainder_len;
            }
        }
    }
    
    // Закрываем канал
    if (close(read_fd) == -1) {
        perror("[Сервер] Ошибка закрытия канала");
        return 1;
    }
    
    printf("[Сервер] Завершен успешно\n");
    return 0;
}

// Функция клиентского процесса
static int client_process(int write_fd) {
    char buffer[1024];
    int string_count = 0;
    
    printf("[Клиент] Начинаю отправку строк в канал...\n");
    
    // Инициализируем генератор случайных чисел
    srand((unsigned int)time(NULL) ^ (unsigned int)getpid());
    
    // Генерируем и отправляем строки случайной длины
    while (1) {
        // Генерируем случайную длину строки [1..100]
        int length = (rand() % MAX_LENGTH) + MIN_LENGTH;
        
        // Создаем строку заданной длины (заполняем случайными символами или просто 'X')
        for (int i = 0; i < length; i++) {
            buffer[i] = 'A' + (rand() % 26); // Случайные буквы
        }
        buffer[length] = '\n';
        buffer[length + 1] = '\0';
        
        // Отправляем строку в канал
        ssize_t bytes_written = write(write_fd, buffer, length + 1);
        if (bytes_written == -1) {
            // Если канал закрыт сервером, это нормально
            if (errno == EPIPE) {
                printf("[Клиент] Канал закрыт сервером, завершаю работу\n");
                break;
            }
            perror("[Клиент] Ошибка записи в канал");
            close(write_fd);
            return 1;
        }
        
        string_count++;
        printf("[Клиент] Отправлена строка #%d длины %d: %.*s\n", 
               string_count, length, length, buffer);
        
        // Небольшая задержка для наглядности
        usleep(100000); // 100 мс
    }
    
    // Закрываем канал
    if (close(write_fd) == -1) {
        perror("[Клиент] Ошибка закрытия канала");
        return 1;
    }
    
    printf("[Клиент] Завершен успешно (отправлено строк: %d)\n", string_count);
    return 0;
}

int task10_main() {
    pid_t server_pid, client_pid;
    int server_status, client_status;
    int pipe_fd[2]; // pipe_fd[0] - для чтения, pipe_fd[1] - для записи
    
    printf("=== Задание 10: Клиент-сервер с анонимным каналом (pipe) ===\n");
    printf("Создаю анонимный канал (pipe)...\n");
    
    // Создаем анонимный канал (pipe)
    if (pipe(pipe_fd) == -1) {
        perror("Ошибка создания pipe");
        return 1;
    }
    
    printf("Анонимный канал создан успешно (read_fd=%d, write_fd=%d)\n", 
           pipe_fd[0], pipe_fd[1]);
    
    // Создаем серверный процесс
    server_pid = fork();
    if (server_pid == -1) {
        perror("Ошибка создания серверного процесса");
        close(pipe_fd[0]);
        close(pipe_fd[1]);
        return 1;
    }
    
    if (server_pid == 0) {
        // Дочерний процесс - сервер
        // Закрываем дескриптор для записи, он нам не нужен
        close(pipe_fd[1]);
        exit(server_process(pipe_fd[0]));
    }
    
    // Создаем клиентский процесс
    client_pid = fork();
    if (client_pid == -1) {
        perror("Ошибка создания клиентского процесса");
        kill(server_pid, SIGTERM);
        waitpid(server_pid, NULL, 0);
        close(pipe_fd[0]);
        close(pipe_fd[1]);
        return 1;
    }
    
    if (client_pid == 0) {
        // Дочерний процесс - клиент
        // Закрываем дескриптор для чтения, он нам не нужен
        close(pipe_fd[0]);
        exit(client_process(pipe_fd[1]));
    }
    
    // Родительский процесс закрывает оба дескриптора
    // (они больше не нужны, так как используются дочерними процессами)
    close(pipe_fd[0]);
    close(pipe_fd[1]);
    
    // Родительский процесс ждет завершения обоих процессов
    printf("\n[Родительский процесс] Ожидаю завершения сервера и клиента...\n\n");
    
    // Ждем завершения сервера
    if (waitpid(server_pid, &server_status, 0) == -1) {
        perror("Ошибка ожидания завершения сервера");
    } else {
        if (WIFEXITED(server_status)) {
            printf("[Родительский процесс] Сервер завершился с кодом: %d\n", 
                   WEXITSTATUS(server_status));
        }
    }
    
    // Ждем завершения клиента
    if (waitpid(client_pid, &client_status, 0) == -1) {
        perror("Ошибка ожидания завершения клиента");
    } else {
        if (WIFEXITED(client_status)) {
            printf("[Родительский процесс] Клиент завершился с кодом: %d\n", 
                   WEXITSTATUS(client_status));
        }
    }
    
    printf("\n=== Задание 10 завершено ===\n");
    return 0;
}