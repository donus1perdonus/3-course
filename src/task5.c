#include "task5.h"
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>

void task5_main(void) {
    printf("=== Задание 5 ===\n");
    printf("Реализация задания 5\n");
    
    // Получение и вывод идентификаторов процессов
    pid_t pid = getpid();
    pid_t ppid = getppid();
    pid_t pgid = getpgid(0);
    
    printf("1. Идентификатор текущего процесса (PID): %d\n", pid);
    printf("2. Идентификатор родительского процесса (PPID): %d\n", ppid);
    printf("3. Идентификатор группы процессов (PGID): %d\n", pgid);
    
    // Получение и вывод идентификаторов пользователей и групп
    uid_t uid = getuid();
    uid_t euid = geteuid();
    gid_t gid = getgid();
    gid_t egid = getegid();
    
    printf("4. Реальный идентификатор пользователя (UID): %d\n", uid);
    printf("5. Эффективный идентификатор пользователя (EUID): %d\n", euid);
    printf("6. Реальный идентификатор группы (GID): %d\n", gid);
    printf("7. Эффективный идентификатор группы (EGID): %d\n", egid);
}
