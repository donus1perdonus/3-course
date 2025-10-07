#include "task6.h"
#include <stdio.h>
#include <unistd.h>

void task6_main(void) {
    printf("=== Задание 6.1 ===\n");
    printf("Реализация задания 6.1\n");
    
    int pid;
    pid = fork();
    printf("%d\n", pid);
    
    printf("=== Задание 6.2 ===\n");
    printf("Реализация задания 6.2\n");

    fork();
    printf("Hi\n");
    fork();
    printf("Hi\n");

    printf("=== Задание 6.3 ===\n");
    printf("Реализация задания 6.3\n");

    if (fork() || fork())
        fork();
    printf("forked! %d", fork());

    printf("=== Задание 6.4 ===\n");
    printf("Реализация задания 6.4\n");

    if (fork() && (!fork())) {
        if (fork() || fork())
            fork();
    }
    printf("52\n");
}
