#include "openmp.h"
#include <omp.h>
#include <random>
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <fstream>

MatrixMultiplier::MatrixMultiplier(int n) : 
    size(n), A(n, std::vector<double>(n)), 
    B(n, std::vector<double>(n)), 
    BT(n, std::vector<double>(n)),
    C(n, std::vector<double>(n, 0.0)) 
{
    initializeMatrices();
    transposeB();
}

void MatrixMultiplier::initializeMatrices()
{
    // Deterministic, thread-safe initialization without shared RNG
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < size; i++) 
    {
        for (int j = 0; j < size; j++) 
        {
            // simple LCG-based pseudo-random deterministic values per cell
            unsigned long long seedA = 1469598103934665603ull ^ (unsigned long long)(i * 1315423911u + j * 2654435761u);
            unsigned long long seedB = 1099511628211ull ^ (unsigned long long)(i * 40503u + j * 9973u);
            seedA ^= seedA << 13; seedA ^= seedA >> 7; seedA ^= seedA << 17;
            seedB ^= seedB << 13; seedB ^= seedB >> 7; seedB ^= seedB << 17;
            A[i][j] = (seedA % 1000003) / 1000003.0; // [0,1)
            B[i][j] = (seedB % 1000033) / 1000033.0; // [0,1)
        }
    }
}

void MatrixMultiplier::transposeB()
{
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            BT[j][i] = B[i][j];
        }
    }
}

void MatrixMultiplier::multiplyLevel1(int num_threads)
{
    #pragma omp parallel for schedule(static) num_threads(num_threads)
    for (int i = 0; i < size; i++) 
    {
        for (int j = 0; j < size; j++) 
        {
            const double* ai = A[i].data();
            const double* btj = BT[j].data();
            double sum = 0.0;
            for (int k = 0; k < size; k++) 
            {
                sum += ai[k] * btj[k];
            }
            C[i][j] = sum;
        }
    }
}

void MatrixMultiplier::multiplyLevel2(int num_threads)
{
    // Parallelize middle loop across j, keep i sequential
    for (int i = 0; i < size; i++) 
    {
        #pragma omp parallel for schedule(static) num_threads(num_threads)
        for (int j = 0; j < size; j++) 
        {
            const double* ai = A[i].data();
            const double* btj = BT[j].data();
            double sum = 0.0;
            for (int k = 0; k < size; k++) 
            {
                sum += ai[k] * btj[k];
            }
            C[i][j] = sum;
        }
    }
}

void MatrixMultiplier::multiplyLevel3(int num_threads)
{
    // Parallelize inner loop k with reduction per element
    for (int i = 0; i < size; i++) 
    {
        for (int j = 0; j < size; j++) 
        {
            const double* ai = A[i].data();
            const double* btj = BT[j].data();
            double sum = 0.0;
            #pragma omp parallel for reduction(+:sum) schedule(static) num_threads(num_threads)
            for (int k = 0; k < size; k++) 
            {
                sum += ai[k] * btj[k];
            }
            C[i][j] = sum;
        }
    }
}
void MatrixMultiplier::resetResult()
{
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < size; i++) 
    {
        for (int j = 0; j < size; j++) 
        {
            C[i][j] = 0.0;
        }
    }
}

std::vector<std::vector<double>> MatrixMultiplier::getResult() const
{
    return C;
}

bool MatrixMultiplier::verifyResult(const std::vector<std::vector<double>> &reference)
{
    for (int i = 0; i < size; i++) 
    {
        for (int j = 0; j < size; j++) 
        {
            if (std::abs(C[i][j] - reference[i][j]) > 1e-6) 
            {
                return false;
            }
        }
    }
    return true;
}

// Реализация класса PiCalculator
PiCalculator::PiCalculator(double precision, int num_threads) 
    : precision(precision), num_threads(num_threads) 
{
}

double PiCalculator::calculatePi() 
{
    double pi_old = 0.0;
    double pi_new = 0.0;
    long long n_points = 1000; // Начальное количество точек
    
    do {
        pi_old = pi_new;
        
        long long points_in_circle = 0;
        
        #pragma omp parallel for reduction(+:points_in_circle) num_threads(num_threads)
        for (long long i = 0; i < n_points; i++) {
            // Генерируем случайные координаты в квадрате [-1, 1] x [-1, 1]
            double x = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
            double y = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
            
            // Проверяем, попадает ли точка в единичную окружность
            if (x * x + y * y <= 1.0) {
                points_in_circle++;
            }
        }
        
        // Вычисляем π по формуле: π ≈ 4 * (точки в круге) / (общее количество точек)
        pi_new = 4.0 * (double)points_in_circle / n_points;
        
        // Удваиваем количество точек для следующей итерации
        n_points *= 2;
        
        std::cout << "n_points: " << n_points/2 << ", π ≈ " << pi_new 
                  << ", разность: " << std::abs(pi_new - pi_old) << std::endl;
                  
    } while (std::abs(pi_new - pi_old) >= precision);
    
    return pi_new;
}

double PiCalculator::calculatePiIntegration() 
{
    double pi_old = 0.0;
    double pi_new = 0.0;
    long long n_intervals = 1000; // Начальное количество интервалов
    
    do {
        pi_old = pi_new;
        
        double sum = 0.0;
        double h = 1.0 / n_intervals; // Шаг интегрирования
        
        #pragma omp parallel for reduction(+:sum) num_threads(num_threads)
        for (long long i = 0; i < n_intervals; i++) {
            double x = (i + 0.5) * h; // Середина интервала
            sum += 4.0 / (1.0 + x * x); // Интегрируем 4/(1+x²) от 0 до 1
        }
        
        pi_new = sum * h;
        
        // Удваиваем количество интервалов для следующей итерации
        n_intervals *= 2;
        
        std::cout << "n_intervals: " << n_intervals/2 << ", π ≈ " << pi_new 
                  << ", разность: " << std::abs(pi_new - pi_old) << std::endl;
                  
    } while (std::abs(pi_new - pi_old) >= precision);
    
    return pi_new;
}

// Реализация класса Task4Calculator
std::vector<int> Task4Calculator::decomposeIntoSquares(int num)
{
    std::vector<int> result;
    if (num <= 0) {
        return result;
    }
    
    // Используем жадный алгоритм для разложения на сумму квадратов
    // (алгоритм Лагранжа: каждое натуральное число можно представить 
    // как сумму не более 4 квадратов)
    while (num > 0) {
        int root = (int)std::sqrt(num);
        int square = root * root;
        result.push_back(square);
        num -= square;
    }
    
    return result;
}

std::vector<long long> Task4Calculator::findNFibonacciNumbers(int count)
{
    std::vector<long long> fib;
    if (count <= 0) {
        return fib;
    }
    
    fib.reserve(count);
    if (count >= 1) {
        fib.push_back(0);
    }
    if (count >= 2) {
        fib.push_back(1);
    }
    
    for (int i = 2; i < count; i++) {
        fib.push_back(fib[i-1] + fib[i-2]);
    }
    
    return fib;
}

long long Task4Calculator::findNthPrime(int nth)
{
    if (nth <= 0) {
        return -1;
    }
    if (nth == 1) {
        return 2;
    }
    
    int count = 1;
    long long candidate = 3;
    
    while (count < nth) {
        bool is_prime = true;
        long long sqrt_candidate = (long long)std::sqrt(candidate);
        
        for (long long i = 2; i <= sqrt_candidate; i++) {
            if (candidate % i == 0) {
                is_prime = false;
                break;
            }
        }
        
        if (is_prime) {
            count++;
            if (count == nth) {
                return candidate;
            }
        }
        
        candidate += 2; // Переходим к следующему нечётному числу
    }
    
    return candidate;
}

long long Task4Calculator::sumOfDivisors(int num)
{
    if (num <= 0) {
        return 0;
    }
    if (num == 1) {
        return 1;
    }
    
    long long sum = 1 + num; // 1 и само число всегда делители
    
    int sqrt_num = (int)std::sqrt(num);
    
    for (int i = 2; i <= sqrt_num; i++) {
        if (num % i == 0) {
            sum += i;
            if (i != num / i) {
                sum += num / i;
            }
        }
    }
    
    return sum;
}

void Task4Calculator::runAllTasks(const std::string& file_path)
{
    std::ifstream input(file_path);
    if (!input.is_open())
    {
        std::cerr << "Не удалось открыть файл '" << file_path << "'" << std::endl;
        return;
    }

    omp_lock_t file_lock;
    omp_init_lock(&file_lock);

    auto tryReadNumber = [&](int& value) -> bool
    {
        bool has_value = false;
        omp_set_lock(&file_lock);
        if (input >> value)
        {
            has_value = true;
        }
        omp_unset_lock(&file_lock);
        return has_value;
    };

    std::cout << "\nРезультаты задания 4 (источник чисел: " << file_path << "):" << std::endl;
    std::cout << "===========================================" << std::endl;
    std::cout << "Каждое подзадание забирает числа по мере чтения общего файла." << std::endl;

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            int value = 0;
            while (tryReadNumber(value))
            {
                auto squares = decomposeIntoSquares(value);
                #pragma omp critical(output)
                {
                    std::cout << "\n[4.1] n = " << value << " -> ";
                    std::cout << value << " = ";
                    for (size_t i = 0; i < squares.size(); ++i)
                    {
                        int root = static_cast<int>(std::sqrt(squares[i]));
                        std::cout << root << "²";
                        if (i + 1 < squares.size())
                        {
                            std::cout << " + ";
                        }
                    }
                    if (squares.empty())
                    {
                        std::cout << "(нет представления)";
                    }
                    std::cout << std::endl;
                }
            }
        }

        #pragma omp section
        {
            int value = 0;
            while (tryReadNumber(value))
            {
                auto fib = findNFibonacciNumbers(value);
                #pragma omp critical(output)
                {
                    std::cout << "\n[4.2] n = " << value << " -> ";
                    if (fib.empty())
                    {
                        std::cout << "последовательность пуста";
                    }
                    else
                    {
                        std::cout << "первые " << value << " чисел: ";
                        for (size_t i = 0; i < fib.size(); ++i)
                        {
                            std::cout << fib[i];
                            if (i + 1 < fib.size())
                            {
                                std::cout << ", ";
                            }
                        }
                    }
                    std::cout << std::endl;
                }
            }
        }

        #pragma omp section
        {
            int value = 0;
            while (tryReadNumber(value))
            {
                long long prime = findNthPrime(value);
                #pragma omp critical(output)
                {
                    std::cout << "\n[4.3] n = " << value << " -> n-е простое число = " << prime << std::endl;
                }
            }
        }

        #pragma omp section
        {
            int value = 0;
            while (tryReadNumber(value))
            {
                long long sum = sumOfDivisors(value);
                #pragma omp critical(output)
                {
                    std::cout << "\n[4.4] n = " << value << " -> сумма делителей = " << sum << std::endl;
                }
            }
        }
    }

    omp_destroy_lock(&file_lock);
}