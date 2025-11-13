#ifndef UTILS_H
#define UTILS_H

#include "openmp.h"
#include <vector>
#include <string>

double measureTime(void (MatrixMultiplier::*func)(int), MatrixMultiplier& multiplier, int threads);

// Структура для хранения аргументов командной строки
struct CommandLineArgs {
    int task_number;
    std::vector<std::string> flags;
};

// Функция для парсинга аргументов командной строки
CommandLineArgs parseCommandLineArgs(int argc, char** argv);

// Функция для вывода справки
void printHelp();

#endif // UTILS_H