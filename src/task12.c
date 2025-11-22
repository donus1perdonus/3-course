#include "task12.h"

#define FIFO_NAME_PREFIX "/tmp/joseph_"
#define FIFO_NAME_SUFFIX "_read"
#define PIDS_FILE "/tmp/joseph_pids"

// Генерация имени FIFO для процесса
static void make_fifo_name(pid_t pid, char *fifo_name, size_t size) {
    snprintf(fifo_name, size, "%s%d%s", FIFO_NAME_PREFIX, (int)pid, FIFO_NAME_SUFFIX);
}

// Функция дочернего процесса в кольце
static int child_process_in_ring(int my_index, int n, int k) {
    (void)k;  // Подавляем предупреждение (k передается через начальное сообщение)
    char my_fifo_name[256];
    char next_fifo_name[256];
    int my_read_fd = -1;
    int next_write_fd = -1;
    pid_t my_pid = getpid();
    pid_t next_pid = 0;
    joseph_message_t message;
    ssize_t bytes_read, bytes_written;
    FILE *fp;
    
    // Создаем имя FIFO для чтения
    make_fifo_name(my_pid, my_fifo_name, sizeof(my_fifo_name));
    
    printf("[Процесс %d (индекс %d)] Создаю FIFO: %s\n", my_pid, my_index, my_fifo_name);
    
    // Удаляем старый FIFO, если существует
    unlink(my_fifo_name);
    
    // Создаем именованный канал для чтения
    if (mkfifo(my_fifo_name, 0666) == -1) {
        perror("[Процесс] Ошибка создания FIFO");
        return 1;
    }
    
    printf("[Процесс %d] FIFO создан, открываю для чтения...\n", my_pid);
    
    // Открываем свой FIFO для чтения (блокируется до открытия для записи)
    my_read_fd = open(my_fifo_name, O_RDONLY);
    if (my_read_fd == -1) {
        perror("[Процесс] Ошибка открытия FIFO для чтения");
        unlink(my_fifo_name);
        return 1;
    }
    
    printf("[Процесс %d] Мой FIFO открыт для чтения, читаю PID следующего процесса из файла...\n", my_pid);
    
    // Читаем PID следующего процесса из файла
    // Ждем, пока родитель запишет файл
    int wait_count = 0;
    while (wait_count < 1000) {
        fp = fopen(PIDS_FILE, "r");
        if (fp != NULL) {
            // Читаем все PID'ы
            pid_t *pids = malloc(n * sizeof(pid_t));
            if (pids == NULL) {
                fclose(fp);
                close(my_read_fd);
                unlink(my_fifo_name);
                return 1;
            }
            
            for (int i = 0; i < n; i++) {
                if (fscanf(fp, "%d", &pids[i]) != 1) {
                    free(pids);
                    fclose(fp);
                    close(my_read_fd);
                    unlink(my_fifo_name);
                    return 1;
                }
            }
            fclose(fp);
            
            // Определяем PID следующего процесса
            int next_index = (my_index + 1) % n;
            next_pid = pids[next_index];
            
            free(pids);
            break;
        }
        usleep(10000); // 10 мс
        wait_count++;
    }
    
    if (wait_count >= 1000 || next_pid == 0) {
        fprintf(stderr, "[Процесс %d] Ошибка: не удалось прочитать PID следующего процесса\n", my_pid);
        close(my_read_fd);
        unlink(my_fifo_name);
        return 1;
    }
    
    printf("[Процесс %d] PID следующего процесса: %d, открываю его FIFO для записи...\n", my_pid, next_pid);
    
    // Создаем имя FIFO следующего процесса
    make_fifo_name(next_pid, next_fifo_name, sizeof(next_fifo_name));
    
    // Открываем FIFO следующего процесса для записи
    // Ждем, пока следующий процесс создаст свой FIFO
    wait_count = 0;
    while (wait_count < 1000) {
        next_write_fd = open(next_fifo_name, O_WRONLY | O_NONBLOCK);
        if (next_write_fd != -1) {
            // Убираем NONBLOCK после успешного открытия
            close(next_write_fd);
            next_write_fd = open(next_fifo_name, O_WRONLY);
            if (next_write_fd != -1) {
                break;
            }
        }
        usleep(10000); // 10 мс
        wait_count++;
    }
    
    if (next_write_fd == -1) {
        perror("[Процесс] Ошибка открытия FIFO следующего процесса для записи");
        close(my_read_fd);
        unlink(my_fifo_name);
        return 1;
    }
    
    printf("[Процесс %d] FIFO следующего процесса открыт, начинаю работу в кольце\n", my_pid);
    
    // Работаем в кольце, пока не получим сигнал на завершение
    while (1) {
        // Читаем сообщение
        bytes_read = read(my_read_fd, &message, sizeof(joseph_message_t));
        
        if (bytes_read == 0) {
            // Канал закрыт
            printf("[Процесс %d] Канал закрыт, завершаю работу\n", my_pid);
            break;
        }
        
        if (bytes_read == -1) {
            perror("[Процесс] Ошибка чтения сообщения");
            break;
        }
        
        if (bytes_read != sizeof(joseph_message_t)) {
            printf("[Процесс %d] Ошибка: прочитано неполное сообщение\n", my_pid);
            continue;
        }
        
        printf("[Процесс %d] Получено сообщение: counter=%d, sender_pid=%d\n", 
               my_pid, message.counter, message.sender_pid);
        
        // Уменьшаем счетчик
        message.counter--;
        message.sender_pid = my_pid;
        
        // Если счетчик достиг нуля, этот процесс должен завершиться
        if (message.counter == 0) {
            printf("[Процесс %d] Счетчик достиг нуля! Завершаюсь...\n", my_pid);
            
            // Закрываем каналы
            close(my_read_fd);
            close(next_write_fd);
            
            // Удаляем свой FIFO
            unlink(my_fifo_name);
            
            // Завершаемся
            exit(0);
        }
        
        // Передаем сообщение следующему процессу
        printf("[Процесс %d] Передаю сообщение следующему процессу (%d): counter=%d\n", 
               my_pid, next_pid, message.counter);
        
        bytes_written = write(next_write_fd, &message, sizeof(joseph_message_t));
        if (bytes_written == -1) {
            // Если следующий процесс уже завершился (EPIPE), ищем следующего живого процесса
            if (errno == EPIPE) {
                printf("[Процесс %d] Следующий процесс (%d) завершен, ищу следующего живого...\n", my_pid, next_pid);
                
                // Читаем все PID'ы из файла
                fp = fopen(PIDS_FILE, "r");
                if (fp != NULL) {
                    pid_t *pids = malloc(n * sizeof(pid_t));
                    if (pids != NULL) {
                        for (int i = 0; i < n; i++) {
                            fscanf(fp, "%d", &pids[i]);
                        }
                        fclose(fp);
                        
                        // Находим свой индекс
                        int my_idx = -1;
                        for (int i = 0; i < n; i++) {
                            if (pids[i] == my_pid) {
                                my_idx = i;
                                break;
                            }
                        }
                        
                        if (my_idx != -1) {
                            // Ищем следующего живого процесса
                            int found = 0;
                            for (int offset = 2; offset < n && !found; offset++) {
                                int next_idx = (my_idx + offset) % n;
                                pid_t candidate_pid = pids[next_idx];
                                
                                // Проверяем, жив ли процесс (используем kill с сигналом 0)
                                if (kill(candidate_pid, 0) == 0) {
                                    // Процесс жив, используем его
                                    close(next_write_fd);
                                    next_pid = candidate_pid;
                                    make_fifo_name(next_pid, next_fifo_name, sizeof(next_fifo_name));
                                    
                                    // Открываем FIFO нового следующего процесса
                                    wait_count = 0;
                                    while (wait_count < 100) {
                                        next_write_fd = open(next_fifo_name, O_WRONLY | O_NONBLOCK);
                                        if (next_write_fd != -1) {
                                            close(next_write_fd);
                                            next_write_fd = open(next_fifo_name, O_WRONLY);
                                            if (next_write_fd != -1) {
                                                found = 1;
                                                printf("[Процесс %d] Найден следующий живой процесс: %d\n", my_pid, next_pid);
                                                break;
                                            }
                                        }
                                        usleep(10000);
                                        wait_count++;
                                    }
                                }
                            }
                            
                            if (!found) {
                                printf("[Процесс %d] Не найден следующий живой процесс, завершаю работу\n", my_pid);
                                free(pids);
                                break;
                            }
                            
                            free(pids);
                            
                            // Повторяем попытку записи
                            bytes_written = write(next_write_fd, &message, sizeof(joseph_message_t));
                            if (bytes_written == -1) {
                                perror("[Процесс] Ошибка записи сообщения после поиска живого процесса");
                                break;
                            }
                        } else {
                            free(pids);
                            break;
                        }
                    } else {
                        fclose(fp);
                        break;
                    }
                } else {
                    printf("[Процесс %d] Не удалось прочитать файл с PID'ами\n", my_pid);
                    break;
                }
            } else {
                perror("[Процесс] Ошибка записи сообщения");
                break;
            }
        }
        
        if (bytes_written != sizeof(joseph_message_t)) {
            printf("[Процесс %d] Ошибка: записано неполное сообщение\n", my_pid);
            break;
        }
    }
    
    // Закрываем каналы
    close(my_read_fd);
    if (next_write_fd != -1) {
        close(next_write_fd);
    }
    
    // Удаляем свой FIFO
    unlink(my_fifo_name);
    
    printf("[Процесс %d] Завершен\n", my_pid);
    return 0;
}

int task12_main(int argc, char* argv[]) {
    int n, k;
    pid_t *child_pids = NULL;
    int *child_statuses = NULL;
    pid_t *termination_order = NULL;
    int termination_count = 0;
    char first_fifo_name[256];
    int first_write_fd = -1;
    joseph_message_t initial_message;
    ssize_t bytes_written;
    FILE *fp;
    
    // Проверяем аргументы командной строки
    // Учитываем, что функция вызывается из main.c, где argv[1] - номер задания
    if (argc < 4) {
        fprintf(stderr, "Использование: %s 12 <n> <k>\n", argv[0]);
        fprintf(stderr, "  n - количество процессов в кольце (натуральное, > 1)\n");
        fprintf(stderr, "  k - количество переходов перед остановкой (натуральное, > 1 и < n)\n");
        return 1;
    }
    
    n = atoi(argv[2]);
    k = atoi(argv[3]);
    
    // Проверяем корректность параметров
    if (n <= 1) {
        fprintf(stderr, "Ошибка: n должно быть натуральным числом больше 1\n");
        return 1;
    }
    
    if (k <= 1 || k >= n) {
        fprintf(stderr, "Ошибка: k должно быть натуральным числом больше 1 и меньше n\n");
        return 1;
    }
    
    printf("=== Задание 12: Задача Иосифа Флавия ===\n");
    printf("Параметры: n = %d, k = %d\n", n, k);
    printf("Создаю %d процессов в кольце...\n", n);
    
    // Выделяем память для массивов
    child_pids = malloc(n * sizeof(pid_t));
    child_statuses = malloc(n * sizeof(int));
    termination_order = malloc(n * sizeof(pid_t));
    
    if (!child_pids || !child_statuses || !termination_order) {
        perror("Ошибка выделения памяти");
        free(child_pids);
        free(child_statuses);
        free(termination_order);
        return 1;
    }
    
    // Удаляем старый файл с PID'ами, если существует
    unlink(PIDS_FILE);
    
    // Создаем n дочерних процессов
    for (int i = 0; i < n; i++) {
        pid_t pid = fork();
        
        if (pid == -1) {
            perror("Ошибка создания дочернего процесса");
            // Убиваем уже созданные процессы
            for (int j = 0; j < i; j++) {
                kill(child_pids[j], SIGTERM);
            }
            // Ждем завершения
            for (int j = 0; j < i; j++) {
                waitpid(child_pids[j], NULL, 0);
            }
            free(child_pids);
            free(child_statuses);
            free(termination_order);
            unlink(PIDS_FILE);
            return 1;
        }
        
        if (pid == 0) {
            // Дочерний процесс
            // Передаем индекс через переменную окружения
            char idx_str[32], n_str[32], k_str[32];
            snprintf(idx_str, sizeof(idx_str), "%d", i);
            snprintf(n_str, sizeof(n_str), "%d", n);
            snprintf(k_str, sizeof(k_str), "%d", k);
            setenv("JOSEPH_INDEX", idx_str, 1);
            setenv("JOSEPH_N", n_str, 1);
            setenv("JOSEPH_K", k_str, 1);
            
            // Запускаем логику дочернего процесса
            exit(child_process_in_ring(i, n, k));
        } else {
            // Родительский процесс
            child_pids[i] = pid;
        }
    }
    
    // Небольшая задержка, чтобы все процессы успели создать свои FIFO
    usleep(200000); // 200 мс
    
    // Записываем массив PID'ов в файл
    fp = fopen(PIDS_FILE, "w");
    if (fp == NULL) {
        perror("Ошибка создания файла с PID'ами");
        for (int i = 0; i < n; i++) {
            kill(child_pids[i], SIGTERM);
        }
        for (int i = 0; i < n; i++) {
            waitpid(child_pids[i], NULL, 0);
        }
        free(child_pids);
        free(child_statuses);
        free(termination_order);
        return 1;
    }
    
    for (int i = 0; i < n; i++) {
        fprintf(fp, "%d\n", (int)child_pids[i]);
    }
    fclose(fp);
    
    printf("[Родитель] Файл с PID'ами создан. Жду, пока процессы прочитают его...\n");
    
    // Небольшая задержка, чтобы процессы успели прочитать PID'ы
    usleep(200000); // 200 мс
    
    // Открываем FIFO первого процесса для записи и отправляем начальное сообщение
    make_fifo_name(child_pids[0], first_fifo_name, sizeof(first_fifo_name));
    
    printf("[Родитель] Открываю FIFO первого процесса (%d) для отправки начального сообщения...\n", child_pids[0]);
    
    first_write_fd = open(first_fifo_name, O_WRONLY);
    if (first_write_fd == -1) {
        perror("[Родитель] Ошибка открытия FIFO первого процесса");
        // Убиваем все процессы
        for (int i = 0; i < n; i++) {
            kill(child_pids[i], SIGTERM);
        }
        for (int i = 0; i < n; i++) {
            waitpid(child_pids[i], NULL, 0);
        }
        free(child_pids);
        free(child_statuses);
        free(termination_order);
        unlink(PIDS_FILE);
        return 1;
    }
    
    // Отправляем начальное сообщение первому процессу
    initial_message.counter = k;
    initial_message.sender_pid = getpid();
    
    printf("[Родитель] Отправляю начальное сообщение первому процессу: counter=%d\n", initial_message.counter);
    
    bytes_written = write(first_write_fd, &initial_message, sizeof(joseph_message_t));
    if (bytes_written == -1) {
        perror("[Родитель] Ошибка записи начального сообщения");
        close(first_write_fd);
        for (int i = 0; i < n; i++) {
            kill(child_pids[i], SIGTERM);
        }
        for (int i = 0; i < n; i++) {
            waitpid(child_pids[i], NULL, 0);
        }
        free(child_pids);
        free(child_statuses);
        free(termination_order);
        unlink(PIDS_FILE);
        return 1;
    }
    
    close(first_write_fd);
    
    printf("[Родитель] Начальное сообщение отправлено. Ожидаю завершения всех процессов...\n\n");
    
    // Ждем завершения всех дочерних процессов и отслеживаем порядок
    for (int i = 0; i < n; i++) {
        pid_t terminated_pid = waitpid(-1, &child_statuses[i], 0);
        if (terminated_pid == -1) {
            perror("[Родитель] Ошибка ожидания завершения процесса");
            continue;
        }
        
        termination_order[termination_count++] = terminated_pid;
        printf("[Родитель] Процесс %d завершен (%d/%d)\n", terminated_pid, termination_count, n);
    }
    
    // Выводим результаты
    printf("\n=== Результаты ===\n");
    printf("n = %d\n", n);
    printf("k = %d\n", k);
    printf("Порядок завершения процессов:\n");
    for (int i = 0; i < termination_count; i++) {
        printf("  %d. PID: %d\n", i + 1, (int)termination_order[i]);
    }
    
    // Удаляем оставшиеся FIFO (на всякий случай)
    for (int i = 0; i < n; i++) {
        char fifo_name[256];
        make_fifo_name(child_pids[i], fifo_name, sizeof(fifo_name));
        unlink(fifo_name);
    }
    
    // Очистка
    free(child_pids);
    free(child_statuses);
    free(termination_order);
    unlink(PIDS_FILE);
    
    printf("\n=== Задание 12 завершено ===\n");
    return 0;
}
