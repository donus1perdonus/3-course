#include <iostream>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <windows.h>
#include "openmp.h"
#include "opencl.h"
#include "utils.h"

void runTask1(const std::vector<std::string>& flags) 
{
    int MATRIX_SIZE = 1000;
    
    // Парсим размер матрицы из флагов
    if (!flags.empty()) {
        try {
            MATRIX_SIZE = std::max(1, std::stoi(flags[0]));
        } catch (...) {
            std::cerr << "Ошибка: неверный размер матрицы. Используется значение по умолчанию (1000)." << std::endl;
        }
    }
    
    std::vector<int> thread_counts = {1, 2, 4, 8, 16};
    
    std::cout << "Задание 1: Перемножение матриц с OpenMP [" << MATRIX_SIZE << "x" << MATRIX_SIZE << "]" << std::endl;
    std::cout << "===========================================" << std::endl;
    
    MatrixMultiplier multiplier(MATRIX_SIZE);
    
    // Создаем эталонный результат с 1 потоком
    std::cout << "Вычисляем эталонный результат с 1 потоком..." << std::endl;
    multiplier.multiplyLevel1(1);
    auto reference = multiplier.getResult();
    
    // Тестируем разные уровни параллелизма
    std::vector<std::string> levels = {"Уровень 1 (внешний)", "Уровень 2 (средний)", "Уровень 3 (внутренний)"};
    std::vector<void (MatrixMultiplier::*)(int)> functions = {
        &MatrixMultiplier::multiplyLevel1,
        &MatrixMultiplier::multiplyLevel2,
        &MatrixMultiplier::multiplyLevel3
    };
    
    for (size_t level_idx = 0; level_idx < levels.size(); level_idx++) {
        std::cout << "\n" << levels[level_idx] << " цикл параллелизация:" << std::endl;
        std::cout << "Потоки\tВремя (с)\tУскорение\tКорректно" << std::endl;
        std::cout << "-------\t--------\t--------\t--------" << std::endl;
        
        double single_thread_time = 0.0;
        
        for (int threads : thread_counts) {
            multiplier.resetResult();
            
            double time = measureTime(functions[level_idx], multiplier, threads);
            bool correct = multiplier.verifyResult(reference);
            double speedup = (threads == 1) ? 1.0 : single_thread_time / time;
            
            if (threads == 1) {
                single_thread_time = time;
            }
            
            std::cout << threads << "\t" << time << "\t" << speedup << "\t\t" 
                      << (correct ? "Да" : "Нет") << std::endl;
        }
    }
}

void runTask2(const std::vector<std::string>& flags) 
{
    double precision = 0.0001;
    int num_threads = 4;
    bool use_monte_carlo = true;
    
    // Парсим флаги
    for (size_t i = 0; i < flags.size(); i++) {
        if (flags[i] == "--monte-carlo") {
            use_monte_carlo = true;
        } else if (flags[i] == "--integration") {
            use_monte_carlo = false;
        } else if (flags[i] == "--help" || flags[i] == "-h") {
            printHelp();
            return;
        } else {
            // Пытаемся парсить как число
            try {
                double value = std::stod(flags[i]);
                if (i == 0) {
                    precision = value;
                } else if (i == 1) {
                    num_threads = std::max(1, (int)value);
                }
            } catch (...) {
                std::cerr << "Предупреждение: неопознанный флаг '" << flags[i] << "'" << std::endl;
            }
        }
    }
    
    std::cout << "Задание 2: Вычисление числа π с точностью " << precision << std::endl;
    std::cout << "Метод: " << (use_monte_carlo ? "Монте-Карло" : "Численное интегрирование") << std::endl;
    std::cout << "Потоков: " << num_threads << std::endl;
    std::cout << "===========================================" << std::endl;
    
    // Инициализируем генератор случайных чисел
    srand(time(nullptr));
    
    PiCalculator calculator(precision, num_threads);
    
    auto start = std::chrono::high_resolution_clock::now();
    double pi_result = use_monte_carlo ? calculator.calculatePi() : calculator.calculatePiIntegration();
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> duration = end - start;
    
    std::cout << "\nРезультат:" << std::endl;
    std::cout << "π ≈ " << pi_result << std::endl;
    std::cout << "Точное значение π = " << M_PI << std::endl;
    std::cout << "Погрешность: " << std::abs(pi_result - M_PI) << std::endl;
    std::cout << "Время выполнения: " << duration.count() << " секунд" << std::endl;
}

void runTask3(const std::vector<std::string>& flags) 
{
    std::string task = "3.1";
    int matrix_size = 1000;
    
    // Парсим флаги
    if (!flags.empty()) {
        // Проверяем, указано ли подзадание
        if (flags[0] == "3.1") {
            printGPUInfo();
            return;
        } else if (flags[0] == "3.2") {
            task = "3.2";
            // Проверяем, указан ли размер матрицы
            if (flags.size() > 1) {
                try {
                    matrix_size = std::max(1, std::stoi(flags[1]));
                } catch (...) {
                    std::cerr << "Ошибка: неверный размер матрицы. Используется значение по умолчанию (1000)." << std::endl;
                }
            }
            multiplyMatricesGPU(matrix_size);
            return;
        } else {
            // Если первый аргумент число, считаем это размером матрицы для 3.2
            try {
                matrix_size = std::max(1, std::stoi(flags[0]));
                task = "3.2";
            } catch (...) {
                // Если не число, используем по умолчанию 3.1
            }
        }
    }
    
    if (task == "3.1") {
        printGPUInfo();
    } else {
        multiplyMatricesGPU(matrix_size);
    }
}

void runTask4(const std::vector<std::string>& flags)
{
    int n = 10; // Значение по умолчанию
    
    // Парсим флаги
    if (!flags.empty()) {
        try {
            n = std::max(1, std::stoi(flags[0]));
        } catch (...) {
            std::cerr << "Ошибка: неверное значение n. Используется значение по умолчанию (10)." << std::endl;
        }
    }
    
    std::cout << "Задание 4: Использование OpenMP sections" << std::endl;
    std::cout << "===========================================" << std::endl;
    std::cout << "Выполнение подзаданий 4.1-4.4 параллельно для n = " << n << std::endl;
    
    Task4Calculator calculator(n);
    calculator.runAllTasks(n);
}

int main(int argc, char** argv) 
{
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    CommandLineArgs args = parseCommandLineArgs(argc, argv);
    
    if (args.flags.empty() && argc == 1) {
        printHelp();
        return 0;
    }
    
    switch (args.task_number) {
        case 1:
            runTask1(args.flags);
            break;
        case 2:
            runTask2(args.flags);
            break;
        case 3:
            runTask3(args.flags);
            break;
        case 4:
            runTask4(args.flags);
            break;
        default:
            std::cerr << "Ошибка: неизвестный номер задания " << args.task_number << std::endl;
            printHelp();
            return 1;
    }
    
    return 0;
}