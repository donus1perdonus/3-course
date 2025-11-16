#include "../include/random_walk.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <chrono>

RandomWalk::RandomWalk() 
    : initialPosition(0.0), numberOfSteps(0), currentPosition(0.0), 
      currentStep(0), isRunning(false), isPaused(false),
      rng(std::chrono::steady_clock::now().time_since_epoch().count()),
      uniformDist(0.0, 1.0) {
}

RandomWalk::RandomWalk(double initialPos, int steps, const DiscreteRandomVariable& stepDist)
    : initialPosition(initialPos), numberOfSteps(steps), stepDistribution(stepDist),
      currentPosition(initialPos), currentStep(0), isRunning(false), isPaused(false),
      rng(std::chrono::steady_clock::now().time_since_epoch().count()),
      uniformDist(0.0, 1.0) {
    computeFinalPositionDistribution();
}

void RandomWalk::setInitialPosition(double pos) {
    initialPosition = pos;
    if (!isRunning) {
        currentPosition = pos;
    }
}

void RandomWalk::setNumberOfSteps(int steps) {
    numberOfSteps = steps;
    if (!isRunning) {
        computeFinalPositionDistribution();
    }
}

void RandomWalk::setStepDistribution(const DiscreteRandomVariable& dist) {
    stepDistribution = dist;
    if (!isRunning) {
        computeFinalPositionDistribution();
    }
}

void RandomWalk::start() {
    if (isRunning) return;
    
    currentPosition = initialPosition;
    currentStep = 0;
    isRunning = true;
    isPaused = false;
    
    // Начинаем первый шаг
    if (numberOfSteps > 0) {
        double velocity = selectRandomStep();
        stepVelocity = velocity;
        stepTargetPosition = currentPosition + velocity;
        stepStartPosition = currentPosition;
        stepStartTime = std::chrono::steady_clock::now();
    }
}

void RandomWalk::pause() {
    if (isRunning && !isPaused) {
        isPaused = true;
    }
}

void RandomWalk::resume() {
    if (isRunning && isPaused) {
        isPaused = false;
        stepStartTime = std::chrono::steady_clock::now();
    }
}

void RandomWalk::stop() {
    isRunning = false;
    isPaused = false;
}

void RandomWalk::reset() {
    stop();
    currentPosition = initialPosition;
    currentStep = 0;
}

void RandomWalk::update(double deltaTime) {
    if (!isRunning || isPaused || currentStep >= numberOfSteps) {
        if (currentStep >= numberOfSteps && isRunning) {
            isRunning = false;
        }
        return;
    }
    
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - stepStartTime).count() / 1000.0;
    
    // Если прошла секунда, завершаем текущий шаг
    if (elapsed >= 1.0) {
        currentPosition = stepTargetPosition;
        currentStep++;
        
        if (currentStep < numberOfSteps) {
            // Начинаем следующий шаг
            double velocity = selectRandomStep();
            stepVelocity = velocity;
            stepTargetPosition = currentPosition + velocity;
            stepStartPosition = currentPosition;
            stepStartTime = std::chrono::steady_clock::now();
        } else {
            isRunning = false;
        }
    } else {
        // Интерполируем позицию для плавной анимации
        double t = elapsed; // t от 0 до 1
        currentPosition = stepStartPosition + (stepTargetPosition - stepStartPosition) * t;
    }
}

void RandomWalk::computeFinalPositionDistribution() {
    finalPositionProbabilities.clear();
    computeDistributionRecursive(initialPosition, 1.0, numberOfSteps);
}

void RandomWalk::computeDistributionRecursive(double position, double probability, int remainingSteps) {
    if (remainingSteps == 0) {
        finalPositionProbabilities[position] += probability;
        return;
    }
    
    auto dist = stepDistribution.getDistribution();
    for (const auto& pair : dist) {
        double velocity = pair.first;
        double stepProb = pair.second;
        computeDistributionRecursive(position + velocity, probability * stepProb, remainingSteps - 1);
    }
}

DiscreteRandomVariable RandomWalk::getFinalPositionDistribution() const {
    std::vector<std::pair<double, double>> dist;
    for (const auto& pair : finalPositionProbabilities) {
        dist.push_back(pair);
    }
    return DiscreteRandomVariable(dist);
}

std::string RandomWalk::getFinalPositionDistributionString() const {
    auto rv = getFinalPositionDistribution();
    return rv.toString();
}

double RandomWalk::selectRandomStep() {
    auto dist = stepDistribution.getDistribution();
    if (dist.empty()) return 0.0;
    
    double randomValue = uniformDist(rng);
    double cumulative = 0.0;
    
    for (const auto& pair : dist) {
        cumulative += pair.second;
        if (randomValue <= cumulative) {
            return pair.first;
        }
    }
    
    // На случай ошибки округления возвращаем последнее значение
    return dist.back().first;
}

