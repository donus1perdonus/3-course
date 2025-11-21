# Operating Systems and System Programming (OSSP)

Проект по курсу "Операционные системы и системное программирование".

## Описание

Проект представляет собой набор из 12 заданий по системному программированию на языке C. Каждое задание реализовано в отдельном модуле и может быть запущено через единую точку входа с указанием номера задания.

## Структура проекта

```
ossp/
├── src/                   # Общие исходные файлы заданий
│   └── task1.с - task12.с
├── include/               # Заголовочные файлы
│   └── task1.h - task12.h
├── windows/               # Windows версия (задания 1-4)
│   ├── main.c             
│   └── CMakeLists.txt     # Конфигурация CMake для Windows
├── linux/                 # Linux версия (задания 5-12)
│   ├── main.c             
│   └── CMakeLists.txt     # Конфигурация CMake для Linux
├── build-windows/         # Сборка Windows версии
│   └── bin/
│       └── OSSP_Project_Windows.exe
├── build-linux/           # Сборка Linux версии
│   └── bin/
│       └── OSSP_Project_Linux
├── .vscode/              # Настройки VS Code
│   └── tasks.json        # Задачи сборки (Windows/Linux)
└── README.md             
```

### Запуск

#### Общий синтаксис
./build-dir/bin/OSSP_Project_Windows <номер_задания> [аргументы]

## Технические требования

- **Стандарт C**: C17
- **Компилятор**: GCC (MinGW)
- **Система сборки**: CMake 3.10+
- **Отладчик**: GDB


## Лицензия

См. файл [LICENSE](LICENSE)
