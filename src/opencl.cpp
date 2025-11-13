#include "opencl.h"
#include <iostream>
#include <chrono>
#include <cstring>
#include <cmath>

#ifdef ENABLE_OPENCL

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

static const char* opencl_error_string(cl_int error) {
    switch(error) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP: return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH: return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE: return "CL_MAP_FAILURE";
        case CL_MISALIGNED_SUB_BUFFER_OFFSET: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case CL_COMPILE_PROGRAM_FAILURE: return "CL_COMPILE_PROGRAM_FAILURE";
        case CL_LINKER_NOT_AVAILABLE: return "CL_LINKER_NOT_AVAILABLE";
        case CL_LINK_PROGRAM_FAILURE: return "CL_LINK_PROGRAM_FAILURE";
        case CL_DEVICE_PARTITION_FAILED: return "CL_DEVICE_PARTITION_FAILED";
        case CL_KERNEL_ARG_INFO_NOT_AVAILABLE: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
        case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE: return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR: return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE: return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER: return "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY: return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION: return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST: return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT: return "CL_INVALID_EVENT";
        case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION";
        case CL_INVALID_GL_OBJECT: return "CL_INVALID_GL_OBJECT";
        case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_MIP_LEVEL: return "CL_INVALID_MIP_LEVEL";
        case CL_INVALID_GLOBAL_WORK_SIZE: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case CL_INVALID_PROPERTY: return "CL_INVALID_PROPERTY";
        default: return "UNKNOWN ERROR";
    }
}

// OpenCL Kernel для умножения матриц
const char* matrix_multiply_kernel = R"(
__kernel void matrix_multiply(__global double* A,
                               __global double* B,
                               __global double* C,
                               int N)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    if (i < N && j < N) {
        double sum = 0.0;
        for (int k = 0; k < N; k++) {
            sum += A[i * N + k] * B[k * N + j];
        }
        C[i * N + j] = sum;
    }
}
)";

void printGPUInfo()
{
    std::cout << "Задание 3.1: Информация о подключённых устройствах" << std::endl;
    std::cout << "=====================================================" << std::endl;
    
    cl_uint num_platforms;
    cl_int err = clGetPlatformIDs(0, NULL, &num_platforms);
    
    if (err != CL_SUCCESS || num_platforms == 0) {
        std::cout << "OpenCL не найден или нет доступных платформ." << std::endl;
        std::cout << "Ошибка: " << opencl_error_string(err) << " (код: " << err << ")" << std::endl;
        return;
    }
    
    cl_platform_id* platforms = new cl_platform_id[num_platforms];
    clGetPlatformIDs(num_platforms, platforms, NULL);
    
    std::cout << "Найдено платформ: " << num_platforms << std::endl << std::endl;
    
    for (cl_uint i = 0; i < num_platforms; i++) {
        char platform_name[1024];
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
        
        std::cout << "Платформа " << i << ": " << platform_name << std::endl;
        
        // Получаем количество устройств GPU на этой платформе
        cl_uint num_devices;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
        
        if (err == CL_SUCCESS && num_devices > 0) {
            cl_device_id* devices = new cl_device_id[num_devices];
            clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);
            
            std::cout << "  GPU устройств: " << num_devices << std::endl;
            
            for (cl_uint j = 0; j < num_devices; j++) {
                char device_name[1024];
                char vendor_name[1024];
                char device_version[1024];
                cl_ulong global_mem_size;
                cl_ulong local_mem_size;
                size_t max_work_group_size;
                cl_uint max_compute_units;
                
                clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
                clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, sizeof(vendor_name), vendor_name, NULL);
                clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, sizeof(device_version), device_version, NULL);
                clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem_size), &global_mem_size, NULL);
                clGetDeviceInfo(devices[j], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem_size), &local_mem_size, NULL);
                clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, NULL);
                clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(max_compute_units), &max_compute_units, NULL);
                
                std::cout << std::endl;
                std::cout << "    Устройство " << j << ":" << std::endl;
                std::cout << "      Имя: " << device_name << std::endl;
                std::cout << "      Производитель: " << vendor_name << std::endl;
                std::cout << "      Версия OpenCL: " << device_version << std::endl;
                std::cout << "      Глобальная память: " << global_mem_size / (1024 * 1024) << " MB" << std::endl;
                std::cout << "      Локальная память: " << local_mem_size / 1024 << " KB" << std::endl;
                std::cout << "      Макс. размер рабочей группы: " << max_work_group_size << std::endl;
                std::cout << "      Вычислительных блоков: " << max_compute_units << std::endl;
            }
            
            delete[] devices;
        } else {
            std::cout << "  GPU устройств не найдено" << std::endl;
        }
        
        std::cout << std::endl;
    }
    
    delete[] platforms;
}

OpenCLMatrixMultiplier::OpenCLMatrixMultiplier() 
    : initialized(false), context(nullptr), queue(nullptr), program(nullptr), kernel(nullptr)
{
}

OpenCLMatrixMultiplier::~OpenCLMatrixMultiplier()
{
    cleanup();
}

bool OpenCLMatrixMultiplier::initialize()
{
    cl_platform_id platform;
    cl_device_id device;
    cl_int err;
    
    // Получаем платформу
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "Ошибка получения платформы: " << opencl_error_string(err) << std::endl;
        return false;
    }
    
    // Получаем GPU устройство
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        std::cerr << "Ошибка получения GPU устройства: " << opencl_error_string(err) << std::endl;
        return false;
    }
    
    // Создаём контекст
    cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0};
    context = clCreateContext(properties, 1, &device, NULL, NULL, &err);
    if (!context || err != CL_SUCCESS) {
        std::cerr << "Ошибка создания контекста: " << opencl_error_string(err) << std::endl;
        return false;
    }
    
    // Создаём очередь команд
    queue = clCreateCommandQueue((cl_context)context, device, 0, &err);
    if (!queue || err != CL_SUCCESS) {
        std::cerr << "Ошибка создания очереди команд: " << opencl_error_string(err) << std::endl;
        clReleaseContext((cl_context)context);
        return false;
    }
    
    // Создаём программу
    program = clCreateProgramWithSource((cl_context)context, 1, &matrix_multiply_kernel, NULL, &err);
    if (!program || err != CL_SUCCESS) {
        std::cerr << "Ошибка создания программы: " << opencl_error_string(err) << std::endl;
        cleanup();
        return false;
    }
    
    // Компилируем программу
    err = clBuildProgram((cl_program)program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo((cl_program)program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = new char[log_size];
        clGetProgramBuildInfo((cl_program)program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        std::cerr << "Ошибка компиляции программы: " << opencl_error_string(err) << std::endl;
        std::cerr << "Лог компиляции:" << std::endl << log << std::endl;
        delete[] log;
        cleanup();
        return false;
    }
    
    // Создаём kernel
    kernel = clCreateKernel((cl_program)program, "matrix_multiply", &err);
    if (!kernel || err != CL_SUCCESS) {
        std::cerr << "Ошибка создания kernel: " << opencl_error_string(err) << std::endl;
        cleanup();
        return false;
    }
    
    initialized = true;
    return true;
}

void OpenCLMatrixMultiplier::multiplyGPU(const std::vector<std::vector<double>>& A,
                                          const std::vector<std::vector<double>>& B,
                                          std::vector<std::vector<double>>& C,
                                          int size)
{
    if (!initialized) {
        std::cerr << "OpenCL не инициализирован" << std::endl;
        return;
    }
    
    // Подготовка массивов для передачи на GPU
    std::vector<double> A_flat(size * size);
    std::vector<double> B_flat(size * size);
    std::vector<double> C_flat(size * size);
    
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            A_flat[i * size + j] = A[i][j];
            B_flat[i * size + j] = B[i][j];
            C_flat[i * size + j] = 0.0;
        }
    }
    
    cl_int err;
    
    // Создаём буферы памяти на GPU
    cl_mem bufA = clCreateBuffer((cl_context)context, CL_MEM_READ_ONLY, sizeof(double) * size * size, NULL, &err);
    cl_mem bufB = clCreateBuffer((cl_context)context, CL_MEM_READ_ONLY, sizeof(double) * size * size, NULL, &err);
    cl_mem bufC = clCreateBuffer((cl_context)context, CL_MEM_WRITE_ONLY, sizeof(double) * size * size, NULL, &err);
    
    // Копируем данные на GPU
    err = clEnqueueWriteBuffer((cl_command_queue)queue, bufA, CL_TRUE, 0, sizeof(double) * size * size, A_flat.data(), 0, NULL, NULL);
    err |= clEnqueueWriteBuffer((cl_command_queue)queue, bufB, CL_TRUE, 0, sizeof(double) * size * size, B_flat.data(), 0, NULL, NULL);
    
    if (err != CL_SUCCESS) {
        std::cerr << "Ошибка копирования данных на GPU: " << opencl_error_string(err) << std::endl;
        clReleaseMemObject(bufA);
        clReleaseMemObject(bufB);
        clReleaseMemObject(bufC);
        return;
    }
    
    // Устанавливаем аргументы для kernel
    clSetKernelArg((cl_kernel)kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg((cl_kernel)kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg((cl_kernel)kernel, 2, sizeof(cl_mem), &bufC);
    clSetKernelArg((cl_kernel)kernel, 3, sizeof(int), &size);
    
    // Запускаем kernel
    size_t global_size[2] = {(size_t)size, (size_t)size};
    err = clEnqueueNDRangeKernel((cl_command_queue)queue, (cl_kernel)kernel, 2, NULL, global_size, NULL, 0, NULL, NULL);
    
    if (err != CL_SUCCESS) {
        std::cerr << "Ошибка запуска kernel: " << opencl_error_string(err) << std::endl;
        clReleaseMemObject(bufA);
        clReleaseMemObject(bufB);
        clReleaseMemObject(bufC);
        return;
    }
    
    // Ожидаем завершения выполнения
    clFinish((cl_command_queue)queue);
    
    // Копируем результат обратно
    err = clEnqueueReadBuffer((cl_command_queue)queue, bufC, CL_TRUE, 0, sizeof(double) * size * size, C_flat.data(), 0, NULL, NULL);
    
    if (err != CL_SUCCESS) {
        std::cerr << "Ошибка копирования результата с GPU: " << opencl_error_string(err) << std::endl;
    }
    
    // Преобразуем обратно в двумерный массив
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            C[i][j] = C_flat[i * size + j];
        }
    }
    
    // Освобождаем память
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
}

void OpenCLMatrixMultiplier::cleanup()
{
    if (kernel) {
        clReleaseKernel((cl_kernel)kernel);
        kernel = nullptr;
    }
    if (program) {
        clReleaseProgram((cl_program)program);
        program = nullptr;
    }
    if (queue) {
        clReleaseCommandQueue((cl_command_queue)queue);
        queue = nullptr;
    }
    if (context) {
        clReleaseContext((cl_context)context);
        context = nullptr;
    }
    initialized = false;
}

void multiplyMatricesGPU(int matrix_size)
{
    std::cout << "Задание 3.2: Умножение матриц на GPU [" << matrix_size << "x" << matrix_size << "]" << std::endl;
    std::cout << "===========================================" << std::endl;
    
    OpenCLMatrixMultiplier gpu_multiplier;
    
    if (!gpu_multiplier.initialize()) {
        std::cout << "Не удалось инициализировать OpenCL" << std::endl;
        return;
    }
    
    std::cout << "Инициализация матриц..." << std::endl;
    
    // Создаём тестовые матрицы
    std::vector<std::vector<double>> A(matrix_size, std::vector<double>(matrix_size));
    std::vector<std::vector<double>> B(matrix_size, std::vector<double>(matrix_size));
    std::vector<std::vector<double>> C(matrix_size, std::vector<double>(matrix_size));
    
    // Инициализируем матрицы (используем ту же логику, что и в OpenMP версии)
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            unsigned long long seedA = 1469598103934665603ull ^ (unsigned long long)(i * 1315423911u + j * 2654435761u);
            unsigned long long seedB = 1099511628211ull ^ (unsigned long long)(i * 40503u + j * 9973u);
            seedA ^= seedA << 13; seedA ^= seedA >> 7; seedA ^= seedA << 17;
            seedB ^= seedB << 13; seedB ^= seedB >> 7; seedB ^= seedB << 17;
            A[i][j] = (seedA % 1000003) / 1000003.0;
            B[i][j] = (seedB % 1000033) / 1000033.0;
        }
    }
    
    std::cout << "Выполняем умножение на GPU..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    gpu_multiplier.multiplyGPU(A, B, C, matrix_size);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> duration = end - start;
    
    std::cout << "Умножение завершено!" << std::endl;
    std::cout << "Время выполнения на GPU: " << duration.count() << " секунд" << std::endl;
    std::cout << std::endl;
    std::cout << "Результат (первые 3x3 элемента):" << std::endl;
    for (int i = 0; i < 3 && i < matrix_size; i++) {
        for (int j = 0; j < 3 && j < matrix_size; j++) {
            std::cout << C[i][j] << "\t";
        }
        std::cout << std::endl;
    }
    
    // Для сравнения выполним на CPU
    std::cout << std::endl;
    std::cout << "Сравнение с CPU вычислениями..." << std::endl;
    
    std::vector<std::vector<double>> C_CPU(matrix_size, std::vector<double>(matrix_size));
    
    auto start_cpu = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            double sum = 0.0;
            for (int k = 0; k < matrix_size; k++) {
                sum += A[i][k] * B[k][j];
            }
            C_CPU[i][j] = sum;
        }
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> duration_cpu = end_cpu - start_cpu;
    
    std::cout << "Время выполнения на CPU: " << duration_cpu.count() << " секунд" << std::endl;
    std::cout << "Ускорение: " << duration_cpu.count() / duration.count() << "x" << std::endl;
    
    // Проверяем корректность
    bool correct = true;
    for (int i = 0; i < matrix_size && correct; i++) {
        for (int j = 0; j < matrix_size; j++) {
            if (std::abs(C[i][j] - C_CPU[i][j]) > 1e-3) {
                correct = false;
                std::cout << "Ошибка в [" << i << "][" << j << "]: GPU=" << C[i][j] << ", CPU=" << C_CPU[i][j] << std::endl;
                break;
            }
        }
    }
    
    std::cout << "Результаты " << (correct ? "корректны" : "некорректны") << std::endl;
}

#else // Если OpenCL не найден

void printGPUInfo() {
    std::cout << "Задание 3.1: Информация о подключённых устройствах" << std::endl;
    std::cout << "=====================================================" << std::endl;
    std::cout << "OpenCL не установлен." << std::endl;
    std::cout << "Для работы с GPU необходимо установить OpenCL SDK." << std::endl;
}

void multiplyMatricesGPU(int matrix_size) {
    std::cout << "Задание 3.2: Умножение матриц на GPU [" << matrix_size << "x" << matrix_size << "]" << std::endl;
    std::cout << "===========================================" << std::endl;
    std::cout << "OpenCL не установлен." << std::endl;
    std::cout << "Для работы с GPU необходимо установить OpenCL SDK." << std::endl;
}

#endif // ENABLE_OPENCL

