#ifndef TASK12_H
#define TASK12_H

#define _DEFAULT_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <signal.h>

// Структура сообщения для передачи по кольцу
typedef struct {
    int counter;      // Счетчик переходов (уменьшается на каждом процессе)
    pid_t sender_pid; // PID отправителя
} joseph_message_t;

// Основная функция задания 12
int task12_main(int argc, char* argv[]);

#endif // TASK12_H
