#include "../include/random_variable.h"
#include <iostream>

DiscreteRandomVariable::DiscreteRandomVariable() : totalProbability(0.0) {}

DiscreteRandomVariable::DiscreteRandomVariable(const std::vector<std::pair<double, double>>& dist) {
    setDistribution(dist);
}

void DiscreteRandomVariable::validateDistribution() {
    if (distribution.empty()) {
        throw std::invalid_argument("Distribution cannot be empty");
    }
    
    // Проверка на уникальность значений
    std::vector<double> values;
    for (const auto& pair : distribution) {
        values.push_back(pair.first);
    }
    std::sort(values.begin(), values.end());
    auto last = std::unique(values.begin(), values.end());
    if (last != values.end()) {
        throw std::invalid_argument("All values must be unique");
    }
    
    // Проверка вероятностей
    totalProbability = 0.0;
    for (const auto& pair : distribution) {
        if (pair.second < 0) {
            throw std::invalid_argument("Probabilities cannot be negative");
        }
        totalProbability += pair.second;
    }
    
    if (std::abs(totalProbability - 1.0) > 1e-10) {
        normalizeProbabilities();
    }
}

void DiscreteRandomVariable::normalizeProbabilities() {
    for (auto& pair : distribution) {
        pair.second /= totalProbability;
    }
    totalProbability = 1.0;
}

void DiscreteRandomVariable::setDistribution(const std::vector<std::pair<double, double>>& dist) {
    distribution = dist;
    validateDistribution();
}

const std::vector<std::pair<double, double>>& DiscreteRandomVariable::getDistribution() const {
    return distribution;
}

DiscreteRandomVariable DiscreteRandomVariable::operator*(double scalar) const {
    std::vector<std::pair<double, double>> result;
    for (const auto& pair : distribution) {
        result.push_back({pair.first * scalar, pair.second});
    }
    return DiscreteRandomVariable(result);
}

DiscreteRandomVariable DiscreteRandomVariable::operator+(const DiscreteRandomVariable& other) const {
    std::map<double, double> resultMap;
    
    for (const auto& pair1 : distribution) {
        for (const auto& pair2 : other.distribution) {
            double sumValue = pair1.first + pair2.first;
            double probProduct = pair1.second * pair2.second;
            resultMap[sumValue] += probProduct;
        }
    }
    
    std::vector<std::pair<double, double>> result;
    for (const auto& pair : resultMap) {
        result.push_back(pair);
    }
    
    return DiscreteRandomVariable(result);
}

DiscreteRandomVariable DiscreteRandomVariable::operator*(const DiscreteRandomVariable& other) const {
    std::map<double, double> resultMap;
    
    for (const auto& pair1 : distribution) {
        for (const auto& pair2 : other.distribution) {
            double productValue = pair1.first * pair2.first;
            double probProduct = pair1.second * pair2.second;
            resultMap[productValue] += probProduct;
        }
    }
    
    std::vector<std::pair<double, double>> result;
    for (const auto& pair : resultMap) {
        result.push_back(pair);
    }
    
    return DiscreteRandomVariable(result);
}

double DiscreteRandomVariable::expectation() const {
    double mean = 0.0;
    for (const auto& pair : distribution) {
        mean += pair.first * pair.second;
    }
    return mean;
}

double DiscreteRandomVariable::variance() const {
    double mean = expectation();
    double var = 0.0;
    for (const auto& pair : distribution) {
        var += std::pow(pair.first - mean, 2) * pair.second;
    }
    return var;
}

double DiscreteRandomVariable::standardDeviation() const {
    return std::sqrt(variance());
}

double DiscreteRandomVariable::skewness() const {
    double mean = expectation();
    double stdDev = standardDeviation();
    if (stdDev == 0) return 0.0;
    
    double skew = 0.0;
    for (const auto& pair : distribution) {
        skew += std::pow((pair.first - mean) / stdDev, 3) * pair.second;
    }
    return skew;
}

double DiscreteRandomVariable::kurtosis() const {
    double mean = expectation();
    double stdDev = standardDeviation();
    if (stdDev == 0) return 0.0;
    
    double kurt = 0.0;
    for (const auto& pair : distribution) {
        kurt += std::pow((pair.first - mean) / stdDev, 4) * pair.second;
    }
    return kurt - 3.0; // Excess kurtosis
}

void DiscreteRandomVariable::serialize(std::ofstream& file) const {
    if (!file.is_open()) {
        throw std::runtime_error("File is not open for writing");
    }
    
    size_t size = distribution.size();
    file.write(reinterpret_cast<const char*>(&size), sizeof(size));
    
    for (const auto& pair : distribution) {
        file.write(reinterpret_cast<const char*>(&pair.first), sizeof(pair.first));
        file.write(reinterpret_cast<const char*>(&pair.second), sizeof(pair.second));
    }
}

void DiscreteRandomVariable::deserialize(std::ifstream& file) {
    if (!file.is_open()) {
        throw std::runtime_error("File is not open for reading");
    }
    
    size_t size;
    file.read(reinterpret_cast<char*>(&size), sizeof(size));
    
    std::vector<std::pair<double, double>> dist;
    for (size_t i = 0; i < size; ++i) {
        double value, probability;
        file.read(reinterpret_cast<char*>(&value), sizeof(value));
        file.read(reinterpret_cast<char*>(&probability), sizeof(probability));
        dist.push_back({value, probability});
    }
    
    setDistribution(dist);
}

std::vector<double> DiscreteRandomVariable::getValues() const {
    std::vector<double> values;
    for (const auto& pair : distribution) {
        values.push_back(pair.first);
    }
    return values;
}

std::vector<double> DiscreteRandomVariable::getProbabilities() const {
    std::vector<double> probs;
    for (const auto& pair : distribution) {
        probs.push_back(pair.second);
    }
    return probs;
}

std::vector<std::pair<double, double>> DiscreteRandomVariable::getCDF() const {
    std::vector<std::pair<double, double>> sorted = distribution;
    std::sort(sorted.begin(), sorted.end(), 
        [](const auto& a, const auto& b) { return a.first < b.first; });
    
    std::vector<std::pair<double, double>> cdf;
    double cumulative = 0.0;
    
    for (const auto& pair : sorted) {
        cumulative += pair.second;
        cdf.push_back({pair.first, cumulative});
    }
    
    return cdf;
}

std::string DiscreteRandomVariable::toString() const {
    std::string result = "Discrete Random Variable:\n";
    for (const auto& pair : distribution) {
        result += "  Value: " + std::to_string(pair.first) + 
                 ", Probability: " + std::to_string(pair.second) + "\n";
    }
    result += "Expectation: " + std::to_string(expectation()) + "\n";
    result += "Variance: " + std::to_string(variance()) + "\n";
    result += "Skewness: " + std::to_string(skewness()) + "\n";
    result += "Kurtosis: " + std::to_string(kurtosis()) + "\n";
    return result;
}