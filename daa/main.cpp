#include <iostream>
#include <chrono>
#include <cmath>
#include "openmp.h"
#include "utils.h"

int main(int argc, char** argv) {
    int MATRIX_SIZE = 5000;
    if (argc > 1) {
        try {
            MATRIX_SIZE = std::max(1, std::stoi(argv[1]));
        } catch (...) {
            // keep default on parse error
        }
    }
    std::vector<int> thread_counts = {1, 2, 4, 8, 16};
    
    std::cout << "Matrix Multiplication Benchmark [" << MATRIX_SIZE << "x" << MATRIX_SIZE << "]" << std::endl;
    std::cout << "===========================================" << std::endl;
    
    MatrixMultiplier multiplier(MATRIX_SIZE);
    
    // Создаем эталонный результат с 1 потоком
    std::cout << "Calculating reference result with 1 thread..." << std::endl;
    multiplier.multiplyLevel1(1);
    auto reference = multiplier.getResult();
    
    // Тестируем разные уровни параллелизма
    std::vector<std::string> levels = {"Level 1 (outer)", "Level 2 (middle)", "Level 3 (inner)"};
    std::vector<void (MatrixMultiplier::*)(int)> functions = {
        &MatrixMultiplier::multiplyLevel1,
        &MatrixMultiplier::multiplyLevel2,
        &MatrixMultiplier::multiplyLevel3
    };
    
    for (size_t level_idx = 0; level_idx < levels.size(); level_idx++) {
        std::cout << "\n" << levels[level_idx] << " loop parallelization:" << std::endl;
        std::cout << "Threads\tTime (s)\tSpeedup\t\tCorrect" << std::endl;
        std::cout << "-------\t--------\t-------\t\t-------" << std::endl;
        
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
                      << (correct ? "Yes" : "No") << std::endl;
        }
    }
    
    return 0;
}