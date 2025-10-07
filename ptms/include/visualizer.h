#ifndef VISUALIZER_H
#define VISUALIZER_H

#include "random_variable.h"
#include <GLFW/glfw3.h>
#include <string>

class Visualizer {
private:
    GLFWwindow* window;
    DiscreteRandomVariable rv;
    int currentView; // 0 - закон распределения, 1 - полилайн, 2 - функция распределения
    
    // Методы отрисовки
    void drawDistributionLaw();
    void drawPolyline();
    void drawCDF();
    void drawAxes();
    void drawText(const std::string& text, double x, double y);
    void drawStatistics();
    
    // Вспомогательные методы
    void setupProjection();
    std::pair<double, double> getValueRange() const;
    std::pair<double, double> getProbabilityRange() const;

public:
    Visualizer();
    ~Visualizer();
    
    bool initialize(int width, int height, const std::string& title);
    void setRandomVariable(const DiscreteRandomVariable& variable);
    void run();
    
    // Метод для получения окна (добавлен для main.cpp)
    GLFWwindow* getWindow() const { return window; }
    
    // Callback functions
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
};

#endif