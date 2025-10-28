#include <iostream>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "openmp.h"
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

int main(int argc, char** argv) 
{
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
            std::cout << "Задание 3: Работа с GPU (CUDA/OpenCL) - пока не реализовано" << std::endl;
            break;
        default:
            std::cerr << "Ошибка: неизвестный номер задания " << args.task_number << std::endl;
            printHelp();
            return 1;
    }
    
    return 0;
}