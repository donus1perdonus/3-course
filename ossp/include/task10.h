#ifndef TASK10_H
#define TASK10_H

#define _DEFAULT_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <errno.h>
#include <time.h>
#include <signal.h>

#define FIFO_NAME "/tmp/task10_fifo"
#define MIN_LENGTH 1
#define MAX_LENGTH 100
#define REQUIRED_COUNT 5

// Основная функция задания 10
int task10_main();

#endif // TASK10_H
