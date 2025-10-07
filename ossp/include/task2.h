#ifndef TASK2_H
#define TASK2_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <ctype.h>

#define MAX_USERS 100
#define MAX_LOGIN_LEN 6
#define MAX_COMMAND_LEN 100
#define MAX_DATETIME_LEN 20

// Структура пользователя
typedef struct {
    char login[MAX_LOGIN_LEN + 1];
    int pin;
    int is_active;
    int max_requests;
    int current_requests;
} User;

// Структура системы
typedef struct {
    User users[MAX_USERS];
    int user_count;
    User* current_user;
} ShellSystem;

// Функции для работы с пользователями
int register_user(ShellSystem* system, const char* login, int pin);
int login_user(ShellSystem* system, const char* login, int pin);
void logout_user(ShellSystem* system);
User* find_user(ShellSystem* system, const char* login);

// Функции команд
void cmd_time(void);
void cmd_date(void);
void cmd_howmuch(const char* datetime, const char* flag);
void cmd_logout(ShellSystem* system);
void cmd_sanctions(ShellSystem* system, const char* username);

// Вспомогательные функции
int is_valid_login(const char* login);
int parse_datetime(const char* datetime_str, struct tm* tm);
void print_help(void);
void shell_loop(ShellSystem* system);
int get_hidden_input(char* buffer, int max_len);

void task2_main(void);

#endif // TASK2_H
