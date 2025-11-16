#ifndef OPENMP_MATRIX_MULTIPLIER_H
#define OPENMP_MATRIX_MULTIPLIER_H

#include <vector>


class MatrixMultiplier 
{
private:

    std::vector<std::vector<double>> A, B, BT, C;
    int size;

public:

    MatrixMultiplier(int n);

    void initializeMatrices();
    void transposeB();

    // Уровень 1: Параллелизация внешнего цикла
    void multiplyLevel1(int num_threads);

    // Уровень 2: Параллелизация среднего цикла
    void multiplyLevel2(int num_threads);

    // Уровень 3: Параллелизация внутреннего цикла
    void multiplyLevel3(int num_threads);

    void resetResult();

    std::vector<std::vector<double>> getResult() const;

    bool verifyResult(const std::vector<std::vector<double>>& reference);
};

// Класс для вычисления числа π с заданной точностью
class PiCalculator 
{
private:
    double precision;
    int num_threads;

public:
    PiCalculator(double precision, int num_threads = 4);
    
    // Вычисление π методом Монте-Карло с удвоением интервалов
    double calculatePi();
    
    // Вычисление π методом численного интегрирования (формула Лейбница)
    double calculatePiIntegration();
};

// Класс для задания 4: использование OpenMP sections
class Task4Calculator 
{
private:
    int n;

public:
    Task4Calculator(int n);
    
    // 4.1: Разложить число n на сумму квадратов
    std::vector<int> decomposeIntoSquares(int num);
    
    // 4.2: Нахождение n чисел Фибоначчи
    std::vector<long long> findNFibonacciNumbers(int count);
    
    // 4.3: Нахождение n-го простого числа
    long long findNthPrime(int nth);
    
    // 4.4: Вывести сумму всех делителей числа n
    long long sumOfDivisors(int num);
    
    // Выполнить все 4 подзадания параллельно с помощью sections
    void runAllTasks(int n_value);
};

#endif // OPENMP_MATRIX_MULTIPLIER_H