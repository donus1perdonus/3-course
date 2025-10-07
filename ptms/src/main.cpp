#include "visualizer.h"
#include "random_variable.h"
#include <iostream>

int main() {
    Visualizer visualizer;
    
    if (!visualizer.initialize(800, 600, "Discrete Random Variable Visualizer")) {
        return -1;
    }
    
    // Устанавливаем указатель на визуализатор для callback'ов
    glfwSetWindowUserPointer(visualizer.getWindow(), &visualizer);
    
    // Демонстрация функциональности
    DiscreteRandomVariable rv1;
    std::vector<std::pair<double, double>> dist1 = {
        {1, 0.1}, {2, 0.2}, {3, 0.4}, {4, 0.2}, {5, 0.1}
    };
    rv1.setDistribution(dist1);
    
    std::cout << "Random Variable 1:" << std::endl;
    std::cout << rv1.toString() << std::endl;
    
    // Умножение на скаляр
    DiscreteRandomVariable rv2 = rv1 * 2.0;
    std::cout << "Random Variable 1 * 2:" << std::endl;
    std::cout << rv2.toString() << std::endl;
    
    // Сложение
    DiscreteRandomVariable rv3 = rv1 + rv1;
    std::cout << "Random Variable 1 + Random Variable 1:" << std::endl;
    std::cout << rv3.toString() << std::endl;
    
    // Установка для визуализации
    visualizer.setRandomVariable(rv1);
    
    std::cout << "\nControls:" << std::endl;
    std::cout << "SPACE - Change view mode" << std::endl;
    std::cout << "ESC - Exit" << std::endl;
    
    visualizer.run();
    
    return 0;
}