#ifndef OPENCL_MATRIX_H
#define OPENCL_MATRIX_H

#include <vector>
#include <string>

#ifdef ENABLE_OPENCL

// Функция для вывода информации о доступных GPU устройствах
void printGPUInfo();

// Функция для умножения матриц на GPU (OpenCL)
void multiplyMatricesGPU(int matrix_size = 1000);

// Класс для работы с матрицами на GPU через OpenCL
class OpenCLMatrixMultiplier
{
private:
    struct DeviceInfo {
        std::string name;
        std::string vendor;
        std::string version;
        size_t max_work_group_size;
        unsigned long long global_mem_size;
        unsigned long long local_mem_size;
        unsigned int max_compute_units;
    };

    std::vector<DeviceInfo> devices;
    bool initialized;
    
    // OpenCL объекты
    void* context;
    void* queue;
    void* program;
    void* kernel;
    std::string kernel_code;
    
public:
    OpenCLMatrixMultiplier();
    ~OpenCLMatrixMultiplier();
    
    bool initialize();
    void printDevices();
    
    // Умножение матриц на GPU
    void multiplyGPU(const std::vector<std::vector<double>>& A,
                      const std::vector<std::vector<double>>& B,
                      std::vector<std::vector<double>>& C,
                      int size);
    
    void cleanup();
};

#else // Если OpenCL не найден

// Заглушки функций
void printGPUInfo();
void multiplyMatricesGPU(int matrix_size);

#endif // ENABLE_OPENCL

#endif // OPENCL_MATRIX_H

