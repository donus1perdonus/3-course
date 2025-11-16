#ifndef RANDOM_WALK_VISUALIZER_H
#define RANDOM_WALK_VISUALIZER_H

#include "random_walk.h"
#include <GLFW/glfw3.h>
#include <string>
#include <chrono>

class RandomWalkVisualizer {
private:
    GLFWwindow* window;
    RandomWalk walk;
    
    // Параметры визуализации
    double viewCenter;
    double viewScale;
    bool showFinalDistribution;
    
    // Для расчета FPS и deltaTime
    std::chrono::steady_clock::time_point lastFrameTime;
    
    // Методы отрисовки
    void drawAxes();
    void drawPoint(double position);
    void drawTrail(const std::vector<std::pair<double, double>>& trail);
    void drawFinalDistribution();
    void drawUI();
    void drawText(const std::string& text, double x, double y);
    
    // Вспомогательные методы
    void setupProjection();
    double screenToWorldX(int screenX);
    int worldToScreenX(double worldX);
    
    // История позиций для отрисовки следа
    std::vector<std::pair<double, double>> positionHistory;
    
public:
    RandomWalkVisualizer();
    ~RandomWalkVisualizer();
    
    bool initialize(int width, int height, const std::string& title);
    void setRandomWalk(const RandomWalk& w);
    void run();
    
    // Управление
    void startWalk();
    void pauseWalk();
    void resumeWalk();
    void stopWalk();
    void resetWalk();
    
    // Настройка параметров
    void setInitialPosition(double pos);
    void setNumberOfSteps(int steps);
    void setStepDistribution(const DiscreteRandomVariable& dist);
    void loadStepDistributionFromFile(const std::string& filename);
    
    // Получение окна
    GLFWwindow* getWindow() const { return window; }
    
    // Получение результата
    RandomWalk& getWalk() { return walk; }
    const RandomWalk& getWalk() const { return walk; }
    
    // Callback functions
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
};

#endif

