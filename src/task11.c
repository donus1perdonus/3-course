#include "task11.h"

// Вычисление НОД (наибольший общий делитель)
static int gcd(int a, int b) {
    while (b != 0) {
        int temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

// Возвращает количество элементов и заполняет массив result
static int compute_reduced_residue_system(int n, int *result, int max_size) {
    int count = 0;
    
    // Приведенная система вычетов: числа от 1 до n-1, взаимно простые с n
    for (int i = 1; i < n && count < max_size; i++) {
        if (gcd(i, n) == 1) {
            result[count++] = i;
        }
    }
    
    return count;
}

// Функция дочернего процесса
static int child_process(int read_fd, int write_fd) {
    int n;
    int reduced_residues[1000]; // Максимальный размер для приведенной системы
    int count;
    ssize_t bytes_read, bytes_written;
    
    // Множество для отслеживания различных значений n, для которых
    // полная и приведенная системы вычетов совпадают
    bool matching_values_seen[MAX_N + 1] = {false};
    int unique_matching_count = 0;
    
    printf("[Дочерний процесс] Начинаю обработку запросов...\n");
    
    while (unique_matching_count < REQUIRED_PRIME_COUNT) {
        // Читаем значение n от родителя
        bytes_read = read(read_fd, &n, sizeof(int));
        
        if (bytes_read == 0) {
            // Канал закрыт родителем - завершаем работу
            printf("[Дочерний процесс] Канал закрыт родителем, завершаю работу\n");
            break;
        }
        
        if (bytes_read == -1) {
            perror("[Дочерний процесс] Ошибка чтения n");
            return 1;
        }
        
        if (bytes_read != sizeof(int)) {
            printf("[Дочерний процесс] Ошибка: прочитано неполное значение\n");
            continue;
        }
        
        printf("[Дочерний процесс] Получено n = %d\n", n);
        
        // Вычисляем приведенную систему вычетов
        count = compute_reduced_residue_system(n, reduced_residues, sizeof(reduced_residues) / sizeof(reduced_residues[0]));
        
        // Проверяем, совпадают ли полная и приведенная системы вычетов
        // Полная система: 0, 1, 2, ..., n-1 (n элементов)
        // Приведенная система: числа от 1 до n-1, взаимно простые с n (count элементов)
        // Они совпадают, если count == n - 1, что означает, что все числа от 1 до n-1 взаимно просты с n
        // Это верно только для простых чисел (или n=1, но n >= MIN_N = 2)
        bool full_and_reduced_match = (count == n - 1);
        
        printf("[Дочерний процесс] Приведенная система вычетов по модулю %d содержит %d элементов, совпадение: %s\n", 
               n, count, full_and_reduced_match ? "ДА" : "НЕТ");
        
        // Если системы совпадают и это новое значение, увеличиваем счетчик
        if (full_and_reduced_match && n >= MIN_N && n <= MAX_N && !matching_values_seen[n]) {
            matching_values_seen[n] = true;
            unique_matching_count++;
            printf("[Дочерний процесс] ✓ Найдено значение с совпадающими системами: %d (уникальных: %d/%d)\n", 
                   n, unique_matching_count, REQUIRED_PRIME_COUNT);
        }
        
        // Отправляем количество элементов
        bytes_written = write(write_fd, &count, sizeof(int));
        if (bytes_written == -1) {
            perror("[Дочерний процесс] Ошибка записи count");
            return 1;
        }
        
        if (bytes_written != sizeof(int)) {
            printf("[Дочерний процесс] Ошибка: записано неполное значение count\n");
            return 1;
        }
        
        // Отправляем элементы приведенной системы вычетов
        if (count > 0) {
            bytes_written = write(write_fd, reduced_residues, count * sizeof(int));
            if (bytes_written == -1) {
                perror("[Дочерний процесс] Ошибка записи элементов");
                return 1;
            }
            
            if (bytes_written != (ssize_t)(count * sizeof(int))) {
                printf("[Дочерний процесс] Ошибка: записано неполное количество элементов\n");
                return 1;
            }
            
            // Выводим первые несколько элементов для наглядности
            printf("[Дочерний процесс] Первые элементы: ");
            int print_count = count < 10 ? count : 10;
            for (int i = 0; i < print_count; i++) {
                printf("%d ", reduced_residues[i]);
            }
            if (count > 10) {
                printf("... (всего %d)", count);
            }
            printf("\n");
        }
    }
    
    printf("[Дочерний процесс] Получено %d различных значений с совпадающими системами. Завершаю работу.\n", 
           unique_matching_count);
    printf("[Дочерний процесс] Завершен успешно\n");
    return 0;
}

// Функция родительского процесса
static int parent_process(int write_fd, int read_fd) {
    int n;
    int count;
    int *reduced_residues = NULL;
    ssize_t bytes_read, bytes_written;
    
    // Инициализируем генератор случайных чисел
    srand((unsigned int)time(NULL) ^ (unsigned int)getpid());
    
    printf("[Родительский процесс] Начинаю отправку значений n...\n");
    
    // Родительский процесс продолжает отправлять значения до тех пор,
    // пока дочерний процесс не завершится (когда он получит 50 различных значений)
    while (1) {
        // Генерируем случайное n в диапазоне [MIN_N, MAX_N]
        n = (rand() % (MAX_N - MIN_N + 1)) + MIN_N;
        
        printf("[Родительский процесс] Отправляю n = %d\n", n);
        
        // Отправляем n дочернему процессу
        bytes_written = write(write_fd, &n, sizeof(int));
        if (bytes_written == -1) {
            // Если канал закрыт дочерним процессом (EPIPE), это нормально
            if (errno == EPIPE) {
                printf("[Родительский процесс] Канал закрыт дочерним процессом, завершаю работу\n");
                break;
            }
            perror("[Родительский процесс] Ошибка записи n");
            free(reduced_residues);
            return 1;
        }
        
        if (bytes_written != sizeof(int)) {
            printf("[Родительский процесс] Ошибка: записано неполное значение n\n");
            free(reduced_residues);
            return 1;
        }
        
        // Читаем количество элементов приведенной системы вычетов
        bytes_read = read(read_fd, &count, sizeof(int));
        if (bytes_read == 0) {
            // Канал закрыт дочерним процессом - он завершил работу
            printf("[Родительский процесс] Канал закрыт дочерним процессом, завершаю работу\n");
            break;
        }
        
        if (bytes_read == -1) {
            perror("[Родительский процесс] Ошибка чтения count");
            free(reduced_residues);
            return 1;
        }
        
        if (bytes_read != sizeof(int)) {
            printf("[Родительский процесс] Ошибка: прочитано неполное значение count\n");
            free(reduced_residues);
            return 1;
        }
        
        // Выделяем память для элементов
        if (count > 0) {
            reduced_residues = realloc(reduced_residues, count * sizeof(int));
            if (reduced_residues == NULL) {
                perror("[Родительский процесс] Ошибка выделения памяти");
                return 1;
            }
            
            // Читаем элементы приведенной системы вычетов
            bytes_read = read(read_fd, reduced_residues, count * sizeof(int));
            if (bytes_read == -1) {
                perror("[Родительский процесс] Ошибка чтения элементов");
                free(reduced_residues);
                return 1;
            }
            
            if (bytes_read != (ssize_t)(count * sizeof(int))) {
                printf("[Родительский процесс] Ошибка: прочитано неполное количество элементов\n");
                free(reduced_residues);
                return 1;
            }
        }
        
        printf("[Родительский процесс] Получена приведенная система вычетов: count = %d\n", count);
    }
    
    free(reduced_residues);
    
    printf("[Родительский процесс] Завершен успешно\n");
    
    // Закрываем канал записи
    close(write_fd);
    
    return 0;
}

int task11_main() {
    pid_t child_pid;
    int child_status;
    int parent_to_child[2];  // pipe[0] - чтение, pipe[1] - запись
    int child_to_parent[2];  // pipe[0] - чтение, pipe[1] - запись
    
    printf("=== Задание 11: Обмен сообщениями через два pipe ===\n");
    
    // Создаем первый pipe (родитель -> дочерний)
    if (pipe(parent_to_child) == -1) {
        perror("Ошибка создания pipe (родитель -> дочерний)");
        return 1;
    }
    printf("Создан pipe: родитель -> дочерний (read_fd=%d, write_fd=%d)\n", 
           parent_to_child[0], parent_to_child[1]);
    
    // Создаем второй pipe (дочерний -> родитель)
    if (pipe(child_to_parent) == -1) {
        perror("Ошибка создания pipe (дочерний -> родитель)");
        close(parent_to_child[0]);
        close(parent_to_child[1]);
        return 1;
    }
    printf("Создан pipe: дочерний -> родитель (read_fd=%d, write_fd=%d)\n", 
           child_to_parent[0], child_to_parent[1]);
    
    // Создаем дочерний процесс
    child_pid = fork();
    if (child_pid == -1) {
        perror("Ошибка создания дочернего процесса");
        close(parent_to_child[0]);
        close(parent_to_child[1]);
        close(child_to_parent[0]);
        close(child_to_parent[1]);
        return 1;
    }
    
    if (child_pid == 0) {
        // Дочерний процесс
        // Закрываем ненужные дескрипторы
        close(parent_to_child[1]);  // не пишем в parent_to_child
        close(child_to_parent[0]);  // не читаем из child_to_parent
        
        exit(child_process(parent_to_child[0], child_to_parent[1]));
    } else {
        // Родительский процесс
        // Закрываем ненужные дескрипторы
        close(parent_to_child[0]);  // не читаем из parent_to_child
        close(child_to_parent[1]);  // не пишем в child_to_parent
        
        // Выполняем логику родительского процесса
        int result = parent_process(parent_to_child[1], child_to_parent[0]);
        
        // Закрываем оставшиеся дескрипторы
        close(parent_to_child[1]);
        close(child_to_parent[0]);
        
        // Ждем завершения дочернего процесса
        printf("\n[Родительский процесс] Ожидаю завершения дочернего процесса...\n");
        if (waitpid(child_pid, &child_status, 0) == -1) {
            perror("Ошибка ожидания завершения дочернего процесса");
        } else {
            if (WIFEXITED(child_status)) {
                printf("[Родительский процесс] Дочерний процесс завершился с кодом: %d\n", 
                       WEXITSTATUS(child_status));
            }
        }
        
        printf("\n=== Задание 11 завершено ===\n");
        return result;
    }
}
