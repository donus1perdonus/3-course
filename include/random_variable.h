#ifndef RANDOM_VARIABLE_H
#define RANDOM_VARIABLE_H

#include <vector>
#include <map>
#include <string>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

class DiscreteRandomVariable {
private:
    std::vector<std::pair<double, double>> distribution; // значение - вероятность
    double totalProbability;

    void validateDistribution();
    void normalizeProbabilities();

public:
    DiscreteRandomVariable();
    DiscreteRandomVariable(const std::vector<std::pair<double, double>>& dist);
    
    // Основные операции
    void setDistribution(const std::vector<std::pair<double, double>>& dist);
    const std::vector<std::pair<double, double>>& getDistribution() const;
    
    // Математические операции
    DiscreteRandomVariable operator*(double scalar) const;
    DiscreteRandomVariable operator+(const DiscreteRandomVariable& other) const;
    DiscreteRandomVariable operator*(const DiscreteRandomVariable& other) const;
    
    // Статистические характеристики
    double expectation() const;
    double variance() const;
    double skewness() const;
    double kurtosis() const;
    double standardDeviation() const;
    
    // Сериализация
    void serialize(std::ofstream& file) const;
    void deserialize(std::ifstream& file);
    
    // Вспомогательные методы для визуализации
    std::vector<double> getValues() const;
    std::vector<double> getProbabilities() const;
    std::vector<std::pair<double, double>> getCDF() const;
    
    // Информация
    std::string toString() const;
};

#endif