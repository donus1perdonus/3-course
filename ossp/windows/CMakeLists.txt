cmake_minimum_required(VERSION 3.10)
project(OSSP_Project_Windows VERSION 1.0.0 LANGUAGES C)

# Установка стандарта C17
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# Опции компилятора для Windows
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -pedantic")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -O0")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3")

# Поиск исходных файлов для Windows (задания 1-4)
file(GLOB_RECURSE SOURCES "*.c" "../src/task1.c" "../src/task2.c" "../src/task3.c" "../src/task4.c")
file(GLOB_RECURSE HEADERS "../include/task1.h" "../include/task2.h" "../include/task3.h" "../include/task4.h")

# Добавление исполняемого файла
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Установка путей для заголовочных файлов
target_include_directories(${PROJECT_NAME} PRIVATE ../include)

# Установка выходной директории
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)

# Windows специфичные настройки
if(WIN32)
    # Дополнительные флаги для Windows
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DWIN32")
    # Для Windows может потребоваться математическая библиотека
    # target_link_libraries(${PROJECT_NAME} m)
endif()

# Информация о проекте
message(STATUS "Project name: ${PROJECT_NAME}")
message(STATUS "Platform: Windows")
message(STATUS "Tasks: 1-4")
message(STATUS "Source files: ${SOURCES}")
message(STATUS "Header files: ${HEADERS}")
