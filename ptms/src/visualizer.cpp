#include "../include/visualizer.h"
#include <iostream>
#include <sstream>
#include <iomanip>

Visualizer::Visualizer() : window(nullptr), currentView(0) {}

Visualizer::~Visualizer() {
    if (window) {
        glfwDestroyWindow(window);
        glfwTerminate();
    }
}

bool Visualizer::initialize(int width, int height, const std::string& title) {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }
    
    window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return false;
    }
    
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    
    // Установка примера распределения
    std::vector<std::pair<double, double>> exampleDist = {
        {1, 0.1}, {2, 0.2}, {3, 0.4}, {4, 0.2}, {5, 0.1}
    };
    rv.setDistribution(exampleDist);
    
    return true;
}

void Visualizer::setRandomVariable(const DiscreteRandomVariable& variable) {
    rv = variable;
}

void Visualizer::run() {
    while (!glfwWindowShouldClose(window)) {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        
        glViewport(0, 0, width, height);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        
        setupProjection();
        
        switch (currentView) {
            case 0:
                drawDistributionLaw();
                break;
            case 1:
                drawPolyline();
                break;
            case 2:
                drawCDF();
                break;
        }
        
        drawAxes();
        drawStatistics();
        
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
}

void Visualizer::setupProjection() {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-0.1, 1.1, -0.1, 1.1, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void Visualizer::drawAxes() {
    glColor3f(0.0f, 0.0f, 0.0f);
    glLineWidth(1.0f);
    
    // Ось X
    glBegin(GL_LINES);
    glVertex2f(0.0f, 0.0f);
    glVertex2f(1.0f, 0.0f);
    glEnd();
    
    // Ось Y
    glBegin(GL_LINES);
    glVertex2f(0.0f, 0.0f);
    glVertex2f(0.0f, 1.0f);
    glEnd();
}

void Visualizer::drawDistributionLaw() {
    auto values = rv.getValues();
    auto probs = rv.getProbabilities();
    
    if (values.empty()) return;
    
    auto valueRange = getValueRange();
    auto probRange = getProbabilityRange();
    
    double minVal = valueRange.first;
    double maxVal = valueRange.second;
    double maxProb = probRange.second;
    
    glColor3f(0.2f, 0.4f, 0.8f);
    glLineWidth(2.0f);
    
    for (size_t i = 0; i < values.size(); ++i) {
        double x = (values[i] - minVal) / (maxVal - minVal) * 0.8 + 0.1;
        double y = (probs[i] / maxProb) * 0.8;
        
        // Вертикальная линия
        glBegin(GL_LINES);
        glVertex2f(x, 0.0f);
        glVertex2f(x, y);
        glEnd();
        
        // Точка сверху
        glPointSize(6.0f);
        glBegin(GL_POINTS);
        glVertex2f(x, y);
        glEnd();
    }
}

void Visualizer::drawPolyline() {
    auto values = rv.getValues();
    auto probs = rv.getProbabilities();
    
    if (values.empty()) return;
    
    auto valueRange = getValueRange();
    auto probRange = getProbabilityRange();
    
    double minVal = valueRange.first;
    double maxVal = valueRange.second;
    double maxProb = probRange.second;
    
    glColor3f(0.8f, 0.2f, 0.2f);
    glLineWidth(2.0f);
    
    glBegin(GL_LINE_STRIP);
    for (size_t i = 0; i < values.size(); ++i) {
        double x = (values[i] - minVal) / (maxVal - minVal) * 0.8 + 0.1;
        double y = (probs[i] / maxProb) * 0.8;
        glVertex2f(x, y);
    }
    glEnd();
    
    glColor3f(0.2f, 0.2f, 0.8f);
    glPointSize(6.0f);
    glBegin(GL_POINTS);
    for (size_t i = 0; i < values.size(); ++i) {
        double x = (values[i] - minVal) / (maxVal - minVal) * 0.8 + 0.1;
        double y = (probs[i] / maxProb) * 0.8;
        glVertex2f(x, y);
    }
    glEnd();
}

void Visualizer::drawCDF() {
    auto cdf = rv.getCDF();
    if (cdf.empty()) return;
    
    auto valueRange = getValueRange();
    double minVal = valueRange.first;
    double maxVal = valueRange.second;
    
    glColor3f(0.2f, 0.6f, 0.2f);
    glLineWidth(2.0f);
    
    glBegin(GL_LINE_STRIP);
    glVertex2f(0.0f, 0.0f);
    
    double prevX = 0.0;
    double prevY = 0.0;
    
    for (size_t i = 0; i < cdf.size(); ++i) {
        double x = (cdf[i].first - minVal) / (maxVal - minVal) * 0.8 + 0.1;
        double y = cdf[i].second * 0.8;
        
        // Горизонтальная линия до текущей точки
        if (i > 0) {
            glVertex2f(x, prevY);
        }
        
        glVertex2f(x, y);
        prevX = x;
        prevY = y;
    }
    
    glVertex2f(1.0f, 1.0f);
    glEnd();
}

void Visualizer::drawText(const std::string& text, double x, double y) {
    // Простая реализация текста с помощью точек
    glColor3f(0.0f, 0.0f, 0.0f);
    glPointSize(1.0f);
    
    // Здесь должна быть более сложная реализация текста
    // Для простоты просто рисуем точку
    glBegin(GL_POINTS);
    glVertex2f(x, y);
    glEnd();
}

void Visualizer::drawStatistics() {
    std::stringstream stats;
    stats << std::fixed << std::setprecision(3);
    
    switch (currentView) {
        case 0:
            stats << "Law of Distribution";
            break;
        case 1:
            stats << "Polyline";
            break;
        case 2:
            stats << "Cumulative Distribution Function";
            break;
    }
    
    stats << "\nE[X] = " << rv.expectation();
    stats << "  Var[X] = " << rv.variance();
    stats << "\nSkew = " << rv.skewness();
    stats << "  Kurt = " << rv.kurtosis();
    stats << "\nPress SPACE to change view";
    
    // В реальном приложении здесь должна быть реализация вывода текста
    // Для простоты выводим в консоль
    std::cout << "\r" << stats.str() << std::flush;
}

std::pair<double, double> Visualizer::getValueRange() const {
    auto values = rv.getValues();
    if (values.empty()) return {0.0, 1.0};
    
    double minVal = *std::min_element(values.begin(), values.end());
    double maxVal = *std::max_element(values.begin(), values.end());
    
    // Добавляем немного отступа
    double range = maxVal - minVal;
    if (range == 0) range = 1.0;
    
    return {minVal - range * 0.1, maxVal + range * 0.1};
}

std::pair<double, double> Visualizer::getProbabilityRange() const {
    auto probs = rv.getProbabilities();
    if (probs.empty()) return {0.0, 1.0};
    
    double maxProb = *std::max_element(probs.begin(), probs.end());
    return {0.0, maxProb * 1.1}; // 10% отступ сверху
}

void Visualizer::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
        Visualizer* visualizer = static_cast<Visualizer*>(glfwGetWindowUserPointer(window));
        
        if (key == GLFW_KEY_SPACE) {
            visualizer->currentView = (visualizer->currentView + 1) % 3;
        } else if (key == GLFW_KEY_ESCAPE) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
    }
}

void Visualizer::framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}