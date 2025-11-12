#include "task9.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <errno.h>

int task9_main() { 
    int pipefd[2];  // Дескрипторы канала: pipefd[0] - чтение, pipefd[1] - запись
    pid_t pid;
    const char *message = "Hello, world!";
    char buffer[256];
    ssize_t bytes_read;
    
    // Создаем неименованный канал
    if (pipe(pipefd) == -1) {
        perror("Ошибка создания канала");
        return 1;
    }
    
    // Создаем дочерний процесс
    pid = fork();
    if (pid == -1) {
        perror("Ошибка создания процесса");
        // Закрываем дескрипторы канала в случае ошибки
        close(pipefd[0]);
        close(pipefd[1]);
        return 1;
    }
    
    if (pid == 0) {
        // Дочерний процесс - читает из канала
        // Закрываем дескриптор записи, так как дочерний процесс только читает
        if (close(pipefd[1]) == -1) {
            perror("Ошибка закрытия дескриптора записи в дочернем процессе");
            close(pipefd[0]);
            exit(1);
        }
        
        // Читаем данные из канала
        bytes_read = read(pipefd[0], buffer, sizeof(buffer) - 1);
        if (bytes_read == -1) {
            perror("Ошибка чтения из канала");
            close(pipefd[0]);
            exit(1);
        }
        
        // Добавляем завершающий нулевой символ
        buffer[bytes_read] = '\0';
        
        // Выводим прочитанную строку в стандартный поток вывода
        printf("Дочерний процесс получил сообщение: %s\n", buffer);
        
        // Закрываем дескриптор чтения
        if (close(pipefd[0]) == -1) {
            perror("Ошибка закрытия дескриптора чтения в дочернем процессе");
            exit(1);
        }
        
        exit(0);
    } else {
        // Родительский процесс - записывает в канал
        // Закрываем дескриптор чтения, так как родительский процесс только записывает
        if (close(pipefd[0]) == -1) {
            perror("Ошибка закрытия дескриптора чтения в родительском процессе");
            close(pipefd[1]);
            return 1;
        }
        
        // Записываем строку в канал
        ssize_t bytes_written = write(pipefd[1], message, strlen(message));
        if (bytes_written == -1) {
            perror("Ошибка записи в канал");
            close(pipefd[1]);
            return 1;
        }
        
        if (bytes_written != (ssize_t)strlen(message)) {
            fprintf(stderr, "Предупреждение: записано не все сообщение (%zd из %zu байт)\n", 
                    bytes_written, strlen(message));
        }
        
        // Закрываем дескриптор записи
        if (close(pipefd[1]) == -1) {
            perror("Ошибка закрытия дескриптора записи в родительском процессе");
            return 1;
        }
        
        // Ждем завершения дочернего процесса
        int status;
        if (waitpid(pid, &status, 0) == -1) {
            perror("Ошибка ожидания завершения дочернего процесса");
            return 1;
        }
        
        if (WIFEXITED(status)) {
            printf("Дочерний процесс завершился с кодом: %d\n", WEXITSTATUS(status));
        } else if (WIFSIGNALED(status)) {
            printf("Дочерний процесс завершился по сигналу: %d\n", WTERMSIG(status));
        }
    }
    
    return 0;
}