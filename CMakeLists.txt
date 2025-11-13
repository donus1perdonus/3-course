cmake_minimum_required(VERSION 3.10)
project(DAA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Прямое указание флагов OpenMP для MSYS2/MINGW
if(MSYS OR MINGW)
    set(OPENMP_FLAGS "-fopenmp")
    set(OPENMP_LIBRARIES "-lgomp")
else()
    set(OPENMP_FLAGS "-fopenmp")
    set(OPENMP_LIBRARIES "")
endif()

# Добавление исполняемого файла
add_executable(DAA
    main.cpp
    src/openmp.cpp
    src/utils.cpp
    src/opencl.cpp
)

target_include_directories(DAA PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Добавление флагов OpenMP
target_compile_options(DAA PRIVATE ${OPENMP_FLAGS})
target_link_libraries(DAA ${OPENMP_LIBRARIES})

# Поиск и добавление OpenCL
find_package(OpenCL QUIET)
if(OpenCL_FOUND)
    target_include_directories(DAA PRIVATE ${OpenCL_INCLUDE_DIRS})
    target_link_libraries(DAA ${OpenCL_LIBRARIES})
    target_compile_definitions(DAA PRIVATE ENABLE_OPENCL)
    message(STATUS "OpenCL найден")
else()
    message(WARNING "OpenCL не найден. GPU функции могут не работать.")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

message(STATUS "OpenMP flags: ${OPENMP_FLAGS}")