#ifndef OPENMP_MATRIX_MULTIPLIER_H
#define OPENMP_MATRIX_MULTIPLIER_H

#include <vector>


class MatrixMultiplier 
{
private:

    std::vector<std::vector<double>> A, B, C;
    int size;

public:

    MatrixMultiplier(int n);

    void initializeMatrices();

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

#endif // OPENMP_MATRIX_MULTIPLIER_H