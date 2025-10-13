#include "openmp.h"
#include <omp.h>
#include <random>
#include <cmath>

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
