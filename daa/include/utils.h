#ifndef UTILS_H
#define UTILS_H

#include "openmp.h"

double measureTime(void (MatrixMultiplier::*func)(int), MatrixMultiplier& multiplier, int threads);

#endif // UTILS_H