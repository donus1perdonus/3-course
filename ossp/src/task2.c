#include "task2.h"

#ifdef _WIN32
#include <conio.h>
#else
#include <termios.h>
#include <unistd.h>
#endif

// Функция для скрытого ввода (не отображает символы на экране)
int get_hidden_input(char* buffer, int max_len) {
    int i = 0;
    char ch;
    
#ifdef _WIN32
    // Для Windows
    while (i < max_len - 1) {
        ch = _getch();
        if (ch == '\r' || ch == '\n') {
            break;
        } else if (ch == '\b' && i > 0) {
            // Backspace
            i--;
            printf("\b \b"); // Стираем символ на экране
        } else if (ch >= 32 && ch <= 126) {
            // Печатные символы
            buffer[i++] = ch;
            printf("*"); // Показываем звездочку вместо символа
        }
    }
#else
    // Для Unix/Linux
    struct termios old_termios, new_termios;
    tcgetattr(STDIN_FILENO, &old_termios);
    new_termios = old_termios;
    new_termios.c_lflag &= ~(ECHO | ICANON);
    tcsetattr(STDIN_FILENO, TCSANOW, &new_termios);
    
    while (i < max_len - 1) {
        ch = getchar();
        if (ch == '\n' || ch == '\r') {
            break;
        } else if (ch == '\b' || ch == 127) {
            // Backspace
            if (i > 0) {
                i--;
                printf("\b \b");
            }
        } else if (ch >= 32 && ch <= 126) {
            buffer[i++] = ch;
            printf("*");
        }
    }
    
    tcsetattr(STDIN_FILENO, TCSANOW, &old_termios);
#endif
    
    buffer[i] = '\0';
    printf("\n");
    return i;
}

// Инициализация системы
void init_system(ShellSystem* system) {
    system->user_count = 0;
    system->current_user = NULL;
    for (int i = 0; i < MAX_USERS; i++) {
        system->users[i].login[0] = '\0';
        system->users[i].pin = 0;
        system->users[i].is_active = 0;
        system->users[i].max_requests = -1; // -1 означает без ограничений
        system->users[i].current_requests = 0;
    }
}

// Проверка валидности логина
int is_valid_login(const char* login) {
    if (strlen(login) == 0 || strlen(login) > MAX_LOGIN_LEN) {
        return 0;
    }
    
    for (int i = 0; login[i]; i++) {
        if (!isalnum(login[i])) {
            return 0;
        }
    }
    return 1;
}

// Регистрация пользователя
int register_user(ShellSystem* system, const char* login, int pin) {
    if (!is_valid_login(login)) {
        printf("Ошибка: неверный формат логина. Логин должен содержать только латинские буквы и цифры, длина до 6 символов.\n");
        return 0;
    }
    
    if (pin < 0 || pin > 100000) {
        printf("Ошибка: PIN должен быть в диапазоне от 0 до 100000.\n");
        return 0;
    }
    
    if (system->user_count >= MAX_USERS) {
        printf("Ошибка: достигнуто максимальное количество пользователей.\n");
        return 0;
    }
    
    // Проверяем, не существует ли уже такой пользователь
    if (find_user(system, login) != NULL) {
        printf("Ошибка: пользователь с таким логином уже существует.\n");
        return 0;
    }
    
    // Добавляем нового пользователя
    User* new_user = &system->users[system->user_count];
    strncpy(new_user->login, login, MAX_LOGIN_LEN);
    new_user->login[MAX_LOGIN_LEN] = '\0';
    new_user->pin = pin;
    new_user->is_active = 1;
    new_user->max_requests = -1;
    new_user->current_requests = 0;
    
    system->user_count++;
    printf("Пользователь %s успешно зарегистрирован.\n", login);
    return 1;
}

// Поиск пользователя по логину
User* find_user(ShellSystem* system, const char* login) {
    for (int i = 0; i < system->user_count; i++) {
        if (strcmp(system->users[i].login, login) == 0) {
            return &system->users[i];
        }
    }
    return NULL;
}

// Авторизация пользователя
int login_user(ShellSystem* system, const char* login, int pin) {
    User* user = find_user(system, login);
    if (user == NULL) {
        printf("Ошибка: пользователь с логином %s не найден.\n", login);
        return 0;
    }
    
    if (user->pin != pin) {
        printf("Ошибка: неверный PIN-код.\n");
        return 0;
    }
    
    if (!user->is_active) {
        printf("Ошибка: пользователь заблокирован.\n");
        return 0;
    }
    
    system->current_user = user;
    user->current_requests = 0; // Сброс счетчика запросов
    printf("Добро пожаловать, %s!\n", login);
    return 1;
}

// Выход из системы
void logout_user(ShellSystem* system) {
    if (system->current_user != NULL) {
        printf("До свидания, %s!\n", system->current_user->login);
        system->current_user = NULL;
    }
}

// Команда Time
void cmd_time(void) {
    time_t now = time(NULL);
    struct tm* local_time = localtime(&now);
    
    printf("Текущее время: %02d:%02d:%02d\n", 
           local_time->tm_hour, 
           local_time->tm_min, 
           local_time->tm_sec);
}

// Команда Date
void cmd_date(void) {
    time_t now = time(NULL);
    struct tm* local_time = localtime(&now);
    
    printf("Текущая дата: %02d:%02d:%04d\n", 
           local_time->tm_mday, 
           local_time->tm_mon + 1, 
           local_time->tm_year + 1900);
}

// Парсинг даты и времени
int parse_datetime(const char* datetime_str, struct tm* tm) {
    // Формат: "дд:ММ:гггг чч:мм:сс"
    int day, month, year, hour, min, sec;
    
    if (sscanf(datetime_str, "%d:%d:%d %d:%d:%d", 
               &day, &month, &year, &hour, &min, &sec) != 6) {
        return 0;
    }
    
    tm->tm_mday = day;
    tm->tm_mon = month - 1;  // tm_mon начинается с 0
    tm->tm_year = year - 1900;  // tm_year - годы с 1900
    tm->tm_hour = hour;
    tm->tm_min = min;
    tm->tm_sec = sec;
    tm->tm_isdst = -1;  // Автоматическое определение летнего времени
    
    return 1;
}

// Команда Howmuch
void cmd_howmuch(const char* datetime, const char* flag) {
    struct tm target_tm;
    if (!parse_datetime(datetime, &target_tm)) {
        printf("Ошибка: неверный формат даты и времени. Используйте формат: дд:ММ:гггг чч:мм:сс\n");
        return;
    }
    
    time_t target_time = mktime(&target_tm);
    if (target_time == -1) {
        printf("Ошибка: неверная дата или время.\n");
        return;
    }
    
    time_t now = time(NULL);
    double diff_seconds = difftime(now, target_time);
    
    if (diff_seconds < 0) {
        printf("Ошибка: указанная дата и время в будущем.\n");
        return;
    }
    
    if (strcmp(flag, "-s") == 0) {
        printf("Прошло секунд: %.0f\n", diff_seconds);
    } else if (strcmp(flag, "-m") == 0) {
        printf("Прошло минут: %.2f\n", diff_seconds / 60.0);
    } else if (strcmp(flag, "-h") == 0) {
        printf("Прошло часов: %.2f\n", diff_seconds / 3600.0);
    } else if (strcmp(flag, "-y") == 0) {
        printf("Прошло лет: %.2f\n", diff_seconds / (365.25 * 24 * 3600));
    } else {
        printf("Ошибка: неверный флаг. Используйте -s, -m, -h или -y\n");
    }
}

// Команда Logout
void cmd_logout(ShellSystem* system) {
    logout_user(system);
}

// Команда Sanctions
void cmd_sanctions(ShellSystem* system, const char* username) {
    User* user = find_user(system, username);
    if (user == NULL) {
        printf("Ошибка: пользователь %s не найден.\n", username);
        return;
    }
    
    printf("Введите количество максимальных запросов для пользователя %s: ", username);
    int max_requests;
    if (scanf("%d", &max_requests) != 1) {
        printf("Ошибка: неверный ввод.\n");
        return;
    }
    
    printf("Для подтверждения введите число 52: ");
    int confirmation;
    if (scanf("%d", &confirmation) != 1 || confirmation != 52) {
        printf("Ошибка: неверное подтверждение. Ограничения не применены.\n");
        return;
    }
    
    user->max_requests = max_requests;
    user->current_requests = 0;
    printf("Ограничения для пользователя %s установлены: максимум %d запросов за сессию.\n", 
           username, max_requests);
}

// Вывод справки
void print_help(void) {
    printf("\nДоступные команды:\n");
    printf("  Time                    - показать текущее время\n");
    printf("  Date                    - показать текущую дату\n");
    printf("  Howmuch <datetime> flag - показать прошедшее время\n");
    printf("                          формат datetime: дд:ММ:гггг чч:мм:сс\n");
    printf("                          флаги: -s (секунды), -m (минуты), -h (часы), -y (годы)\n");
    printf("  Logout                  - выйти из системы\n");
    printf("  Sanctions <username>    - установить ограничения для пользователя\n");
    printf("  Help                    - показать эту справку\n");
    printf("  Exit                    - завершить работу программы\n\n");
}

// Основной цикл оболочки
void shell_loop(ShellSystem* system) {
    char command[MAX_COMMAND_LEN];
    char datetime[MAX_DATETIME_LEN];
    char flag[10];
    char username[MAX_LOGIN_LEN + 1];
    
    printf("\n=== Оболочка командной строки ===\n");
    print_help();
    
    while (1) {
        if (system->current_user == NULL) {
            // Меню авторизации/регистрации
            printf("\n=== Авторизация ===\n");
            printf("1. Войти\n");
            printf("2. Зарегистрироваться\n");
            printf("3. Выход\n");
            printf("Выберите действие: ");
            
            int choice;
            if (scanf("%d", &choice) != 1) {
                printf("Ошибка: неверный ввод.\n");
                continue;
            }
            
            if (choice == 1) {
                printf("Введите логин: ");
                scanf("%6s", username);
                printf("Введите PIN: ");
                char pin_str[10];
                get_hidden_input(pin_str, sizeof(pin_str));
                int pin = atoi(pin_str);
                login_user(system, username, pin);
            } else if (choice == 2) {
                printf("Введите логин (до 6 символов, только буквы и цифры): ");
                scanf("%6s", username);
                printf("Введите PIN (0-100000): ");
                char pin_str[10];
                get_hidden_input(pin_str, sizeof(pin_str));
                int pin = atoi(pin_str);
                register_user(system, username, pin);
            } else if (choice == 3) {
                printf("До свидания!\n");
                break;
            } else {
                printf("Неверный выбор.\n");
            }
        } else {
            // Проверяем ограничения
            if (system->current_user->max_requests != -1 && 
                system->current_user->current_requests >= system->current_user->max_requests) {
                printf("Достигнуто максимальное количество запросов (%d). Выход из системы.\n", 
                       system->current_user->max_requests);
                logout_user(system);
                continue;
            }
            
            printf("\n[%s@shell] $ ", system->current_user->login);
            scanf("%s", command);
            
            // Увеличиваем счетчик запросов
            system->current_user->current_requests++;
            
            if (strcmp(command, "Time") == 0) {
                cmd_time();
            } else if (strcmp(command, "Date") == 0) {
                cmd_date();
            } else if (strcmp(command, "Howmuch") == 0) {
                printf("Введите дату и время (дд:ММ:гггг чч:мм:сс): ");
                scanf(" %19[^\n]", datetime);
                printf("Введите флаг (-s, -m, -h, -y): ");
                scanf("%s", flag);
                cmd_howmuch(datetime, flag);
            } else if (strcmp(command, "Logout") == 0) {
                cmd_logout(system);
            } else if (strcmp(command, "Sanctions") == 0) {
                printf("Введите имя пользователя: ");
                scanf("%6s", username);
                cmd_sanctions(system, username);
            } else if (strcmp(command, "Help") == 0) {
                print_help();
            } else if (strcmp(command, "Exit") == 0) {
                printf("До свидания!\n");
                break;
            } else {
                printf("Неизвестная команда. Введите 'Help' для справки.\n");
            }
        }
    }
}

void task2_main(void) {
    printf("=== Задание 2: Примитивная оболочка командной строки ===\n");
    
    ShellSystem system;
    init_system(&system);
    
    // Добавляем тестового пользователя
    register_user(&system, "admin", 12345);
    
    shell_loop(&system);
}
