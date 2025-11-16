#include "../include/random_walk_visualizer.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

RandomWalkVisualizer::RandomWalkVisualizer() 
    : window(nullptr), viewCenter(0.0), viewScale(10.0), showFinalDistribution(false),
      walk(0.0, 10, DiscreteRandomVariable()) {
    lastFrameTime = std::chrono::steady_clock::now();
}

RandomWalkVisualizer::~RandomWalkVisualizer() {
    if (window) {
        glfwDestroyWindow(window);
        glfwTerminate();
    }
}

bool RandomWalkVisualizer::initialize(int width, int height, const std::string& title) {
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
    
    return true;
}

void RandomWalkVisualizer::setRandomWalk(const RandomWalk& w) {
    walk = w;
    positionHistory.clear();
    positionHistory.push_back({walk.getCurrentPosition(), 0.0});
}

void RandomWalkVisualizer::startWalk() {
    walk.start();
    positionHistory.clear();
    positionHistory.push_back({walk.getCurrentPosition(), 0.0});
}

void RandomWalkVisualizer::pauseWalk() {
    walk.pause();
}

void RandomWalkVisualizer::resumeWalk() {
    walk.resume();
}

void RandomWalkVisualizer::stopWalk() {
    walk.stop();
}

void RandomWalkVisualizer::resetWalk() {
    walk.reset();
    positionHistory.clear();
    positionHistory.push_back({walk.getCurrentPosition(), 0.0});
}

void RandomWalkVisualizer::setInitialPosition(double pos) {
    if (!walk.getIsRunning()) {
        walk.setInitialPosition(pos);
        positionHistory.clear();
        positionHistory.push_back({walk.getCurrentPosition(), 0.0});
    }
}

void RandomWalkVisualizer::setNumberOfSteps(int steps) {
    if (!walk.getIsRunning()) {
        walk.setNumberOfSteps(steps);
    }
}

void RandomWalkVisualizer::setStepDistribution(const DiscreteRandomVariable& dist) {
    if (!walk.getIsRunning()) {
        walk.setStepDistribution(dist);
    }
}

void RandomWalkVisualizer::loadStepDistributionFromFile(const std::string& filename) {
    try {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Не удалось открыть файл: " + filename);
        }
        
        DiscreteRandomVariable rv;
        rv.deserialize(file);
        file.close();
        
        walk.setStepDistribution(rv);
        std::cout << "Распределение загружено из файла: " << filename << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Ошибка при загрузке распределения: " << ex.what() << std::endl;
    }
}

void RandomWalkVisualizer::run() {
    while (!glfwWindowShouldClose(window)) {
        // Расчет deltaTime
        auto currentTime = std::chrono::steady_clock::now();
        auto deltaTime = std::chrono::duration_cast<std::chrono::milliseconds>(
            currentTime - lastFrameTime).count() / 1000.0;
        lastFrameTime = currentTime;
        
        // Обновление состояния блуждания
        walk.update(deltaTime);
        
        // Обновление истории позиций
        double currentPos = walk.getCurrentPosition();
        if (!positionHistory.empty()) {
            double lastPos = positionHistory.back().first;
            if (std::abs(currentPos - lastPos) > 0.01) {
                positionHistory.push_back({currentPos, 0.0});
            }
        } else {
            positionHistory.push_back({currentPos, 0.0});
        }
        
        // Автоматическое центрирование камеры на текущей позиции
        viewCenter = currentPos;
        
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        
        glViewport(0, 0, width, height);
        glClearColor(0.95f, 0.95f, 0.95f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        
        setupProjection();
        
        drawAxes();
        drawTrail(positionHistory);
        drawPoint(currentPos);
        
        if (walk.isCompleted() && showFinalDistribution) {
            drawFinalDistribution();
        }
        
        drawUI();
        
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
}

void RandomWalkVisualizer::setupProjection() {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    
    double left = viewCenter - viewScale;
    double right = viewCenter + viewScale;
    double bottom = -1.0;
    double top = 1.0;
    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(left, right, bottom, top, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void RandomWalkVisualizer::drawAxes() {
    glColor3f(0.3f, 0.3f, 0.3f);
    glLineWidth(1.0f);
    
    // Горизонтальная ось (ℝ)
    glBegin(GL_LINES);
    glVertex2d(viewCenter - viewScale * 2, 0.0);
    glVertex2d(viewCenter + viewScale * 2, 0.0);
    glEnd();
    
    // Вертикальная линия в начале координат
    glBegin(GL_LINES);
    glVertex2d(0.0, -0.1);
    glVertex2d(0.0, 0.1);
    glEnd();
    
    // Разметка на оси
    glColor3f(0.5f, 0.5f, 0.5f);
    glLineWidth(0.5f);
    for (int i = static_cast<int>(viewCenter - viewScale); i <= static_cast<int>(viewCenter + viewScale); i += 1) {
        if (i == 0) continue;
        glBegin(GL_LINES);
        glVertex2d(i, -0.05);
        glVertex2d(i, 0.05);
        glEnd();
    }
}

void RandomWalkVisualizer::drawPoint(double position) {
    glColor3f(1.0f, 0.0f, 0.0f);
    glPointSize(8.0f);
    
    glBegin(GL_POINTS);
    glVertex2d(position, 0.0);
    glEnd();
    
    // Круг вокруг точки
    glColor3f(1.0f, 0.0f, 0.0f);
    glLineWidth(2.0f);
    const int segments = 32;
    glBegin(GL_LINE_LOOP);
    for (int i = 0; i < segments; ++i) {
        double angle = 2.0 * M_PI * i / segments;
        double radius = 0.15;
        glVertex2d(position + radius * cos(angle), radius * sin(angle));
    }
    glEnd();
}

void RandomWalkVisualizer::drawTrail(const std::vector<std::pair<double, double>>& trail) {
    if (trail.size() < 2) return;
    
    glColor3f(0.0f, 0.5f, 1.0f);
    glLineWidth(2.0f);
    
    glBegin(GL_LINE_STRIP);
    for (const auto& point : trail) {
        glVertex2d(point.first, 0.0);
    }
    glEnd();
}

void RandomWalkVisualizer::drawFinalDistribution() {
    auto finalDist = walk.getFinalPositionDistribution();
    auto dist = finalDist.getDistribution();
    
    if (dist.empty()) return;
    
    // Находим максимальную вероятность для масштабирования
    double maxProb = 0.0;
    for (const auto& pair : dist) {
        maxProb = std::max(maxProb, pair.second);
    }
    
    if (maxProb <= 0.0) return;
    
    // Рисуем столбцы распределения
    glColor3f(0.0f, 0.8f, 0.0f);
    glLineWidth(2.0f);
    
    for (const auto& pair : dist) {
        double position = pair.first;
        double prob = pair.second;
        double height = (prob / maxProb) * 0.8; // Масштабируем до 80% высоты
        
        // Столбец
        glBegin(GL_QUADS);
        glVertex2d(position - 0.1, 0.0);
        glVertex2d(position + 0.1, 0.0);
        glVertex2d(position + 0.1, height);
        glVertex2d(position - 0.1, height);
        glEnd();
    }
}

void RandomWalkVisualizer::drawUI() {
    // Информация о состоянии
    std::stringstream ss;
    ss << "Position: " << std::fixed << std::setprecision(2) << walk.getCurrentPosition();
    ss << " | Step: " << walk.getCurrentStep();
    
    if (walk.getIsRunning()) {
        if (walk.getIsPaused()) {
            ss << " | [PAUSED]";
        } else {
            ss << " | [RUNNING]";
        }
    } else if (walk.isCompleted()) {
        ss << " | [COMPLETED]";
    } else {
        ss << " | [STOPPED]";
    }
    
    // Простой текстовый вывод в консоль (для полноценного UI нужна библиотека типа ImGui)
    // Здесь просто выводим в заголовок окна
    std::string title = "Random Walk - " + ss.str();
    glfwSetWindowTitle(window, title.c_str());
}

void RandomWalkVisualizer::drawText(const std::string& text, double x, double y) {
    // Простая реализация через точки (для полноценного текста нужна библиотека)
    // Здесь оставляем заглушку
}

double RandomWalkVisualizer::screenToWorldX(int screenX) {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    double normalizedX = (2.0 * screenX / width) - 1.0;
    return viewCenter + normalizedX * viewScale;
}

int RandomWalkVisualizer::worldToScreenX(double worldX) {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    double normalizedX = (worldX - viewCenter) / viewScale;
    return static_cast<int>((normalizedX + 1.0) * width / 2.0);
}

void RandomWalkVisualizer::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action != GLFW_PRESS) return;
    
    RandomWalkVisualizer* viz = static_cast<RandomWalkVisualizer*>(glfwGetWindowUserPointer(window));
    if (!viz) return;
    
    switch (key) {
        case GLFW_KEY_SPACE:
            if (viz->walk.getIsRunning()) {
                if (viz->walk.getIsPaused()) {
                    viz->resumeWalk();
                } else {
                    viz->pauseWalk();
                }
            } else {
                viz->startWalk();
            }
            break;
        case GLFW_KEY_R:
            viz->resetWalk();
            break;
        case GLFW_KEY_S:
            viz->stopWalk();
            break;
        case GLFW_KEY_D:
            viz->showFinalDistribution = !viz->showFinalDistribution;
            break;
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(window, GLFW_TRUE);
            break;
    }
}

void RandomWalkVisualizer::framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

