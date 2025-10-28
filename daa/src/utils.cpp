#include "utils.h"
#include <chrono>
#include <iostream>
#include <sstream>

double measureTime(void (MatrixMultiplier::*func)(int), MatrixMultiplier& multiplier, int threads) 
{
    auto start = std::chrono::high_resolution_clock::now();
    (multiplier.*func)(threads);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    return duration.count();
}

CommandLineArgs parseCommandLineArgs(int argc, char** argv) 
{
    CommandLineArgs args;
    args.task_number = 1; // По умолчанию
    
    if (argc < 2) {
        return args;
    }
    
    // Первый аргумент - номер задания
    try {
        args.task_number = std::stoi(argv[1]);
    } catch (...) {
        std::cerr << "Ошибка: неверный номер задания. Используется значение по умолчанию (1)." << std::endl;
        args.task_number = 1;
    }
    
    // Остальные аргументы - флаги
    for (int i = 2; i < argc; i++) {
        args.flags.push_back(std::string(argv[i]));
    }
    
    return args;
}

void printHelp() 
{
    std::cout << "Использование: DAA.exe <номер_задания> [флаги...]" << std::endl;
    std::cout << std::endl;
    std::cout << "Задания:" << std::endl;
    std::cout << "  1 - Перемножение матриц с OpenMP" << std::endl;
    std::cout << "  2 - Вычисление числа π с заданной точностью" << std::endl;
    std::cout << "  3 - Работа с GPU (CUDA/OpenCL)" << std::endl;
    std::cout << std::endl;
    std::cout << "Флаги для задания 1:" << std::endl;
    std::cout << "  <размер_матрицы> - размер матрицы (по умолчанию 1000)" << std::endl;
    std::cout << std::endl;
    std::cout << "Флаги для задания 2:" << std::endl;
    std::cout << "  <точность> - требуемая точность вычисления π (по умолчанию 0.0001)" << std::endl;
    std::cout << "  <потоки> - количество потоков OpenMP (по умолчанию 4)" << std::endl;
    std::cout << "  --monte-carlo - использовать метод Монте-Карло (по умолчанию)" << std::endl;
    std::cout << "  --integration - использовать метод численного интегрирования" << std::endl;
    std::cout << std::endl;
    std::cout << "Примеры:" << std::endl;
    std::cout << "  DAA.exe 1 2000" << std::endl;
    std::cout << "  DAA.exe 2 0.001 8 --integration" << std::endl;
    std::cout << "  DAA.exe 2 0.00001 16 --monte-carlo" << std::endl;
}