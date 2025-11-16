#ifndef RANDOM_WALK_H
#define RANDOM_WALK_H

#include "random_variable.h"
#include <vector>
#include <map>
#include <chrono>
#include <random>

class RandomWalk {
private:
    double initialPosition;
    int numberOfSteps;
    DiscreteRandomVariable stepDistribution; // ДСВ для скоростей перемещения
    double currentPosition;
    int currentStep;
    bool isRunning;
    bool isPaused;
    
    std::chrono::steady_clock::time_point stepStartTime;
    double stepStartPosition;
    double stepTargetPosition;
    double stepVelocity;
    
    // Для вычисления распределения конечных позиций
    std::map<double, double> finalPositionProbabilities;
    
    // Генератор случайных чисел для выбора шагов
    std::mt19937 rng;
    std::uniform_real_distribution<double> uniformDist;
    
    // Вычисление распределения конечных позиций
    void computeFinalPositionDistribution();
    void computeDistributionRecursive(double position, double probability, int remainingSteps);
    
    // Выбор случайного шага согласно распределению
    double selectRandomStep();
    
public:
    RandomWalk();
    RandomWalk(double initialPos, int steps, const DiscreteRandomVariable& stepDist);
    
    // Настройка параметров
    void setInitialPosition(double pos);
    void setNumberOfSteps(int steps);
    void setStepDistribution(const DiscreteRandomVariable& dist);
    
    // Управление процессом
    void start();
    void pause();
    void resume();
    void stop();
    void reset();
    
    // Обновление состояния (вызывается каждый кадр)
    void update(double deltaTime);
    
    // Получение состояния
    double getCurrentPosition() const { return currentPosition; }
    int getCurrentStep() const { return currentStep; }
    bool getIsRunning() const { return isRunning; }
    bool getIsPaused() const { return isPaused; }
    bool isCompleted() const { return currentStep >= numberOfSteps && !isRunning; }
    
    // Получение результата
    DiscreteRandomVariable getFinalPositionDistribution() const;
    std::string getFinalPositionDistributionString() const;
};

#endif

