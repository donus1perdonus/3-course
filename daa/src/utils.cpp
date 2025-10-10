#include "utils.h"
#include <chrono>

double measureTime(void (MatrixMultiplier::*func)(int), MatrixMultiplier& multiplier, int threads) 
{
    auto start = std::chrono::high_resolution_clock::now();
    (multiplier.*func)(threads);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    return duration.count();
}