#include "openmp.h"
#include <omp.h>
#include <random>
#include <cmath>

MatrixMultiplier::MatrixMultiplier(int n) : 
    size(n), A(n, std::vector<double>(n)), 
    B(n, std::vector<double>(n)), 
    C(n, std::vector<double>(n, 0.0)) 
{
    initializeMatrices();
}

void MatrixMultiplier::initializeMatrices()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < size; i++) 
    {
        for (int j = 0; j < size; j++) 
        {
            A[i][j] = dis(gen);
            B[i][j] = dis(gen);
        }
    }
}

void MatrixMultiplier::multiplyLevel1(int num_threads)
{
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < size; i++) 
    {
        for (int j = 0; j < size; j++) 
        {
            double sum = 0.0;
            for (int k = 0; k < size; k++) 
            {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

void MatrixMultiplier::multiplyLevel2(int num_threads)
{
    for (int i = 0; i < size; i++) 
    {
        #pragma omp parallel for num_threads(num_threads)
        for (int j = 0; j < size; j++) 
        {
            double sum = 0.0;
            for (int k = 0; k < size; k++) 
            {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

void MatrixMultiplier::multiplyLevel3(int num_threads)
{
    for (int i = 0; i < size; i++) 
    {
        for (int j = 0; j < size; j++) 
        {
            double sum = 0.0;
            #pragma omp parallel for reduction(+:sum) num_threads(num_threads)
            for (int k = 0; k < size; k++) 
            {
                sum += A[i][k] * B[k][j];
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
