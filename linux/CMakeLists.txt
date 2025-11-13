cmake_minimum_required(VERSION 3.10)
project(OSSP_Project_Linux VERSION 1.0.0 LANGUAGES C)

# Установка стандарта C17
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# Опции компилятора для Linux
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -pedantic")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -O0")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3")

# Поиск исходных файлов для Linux (задания 5-12)
file(GLOB_RECURSE SOURCES "*.c" "../src/task5.c" "../src/task6.c" "../src/task7.c" "../src/task8.c" "../src/task9.c" "../src/task10.c" "../src/task11.c" "../src/task12.c")
file(GLOB_RECURSE HEADERS "../include/task5.h" "../include/task6.h" "../include/task7.h" "../include/task8.h" "../include/task9.h" "../include/task10.h" "../include/task11.h" "../include/task12.h")

# Добавление исполняемого файла
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Установка путей для заголовочных файлов
target_include_directories(${PROJECT_NAME} PRIVATE ../include)

# Установка выходной директории
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)

# Linux специфичные настройки
if(UNIX AND NOT APPLE)
    # Дополнительные флаги для Linux
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DLINUX")
    # Для Linux может потребоваться математическая библиотека
    target_link_libraries(${PROJECT_NAME} m)
    # Для Linux может потребоваться библиотека pthread
    find_package(Threads REQUIRED)
    target_link_libraries(${PROJECT_NAME} Threads::Threads)
endif()

# Информация о проекте
message(STATUS "Project name: ${PROJECT_NAME}")
message(STATUS "Platform: Linux")
message(STATUS "Tasks: 5-8")
message(STATUS "Source files: ${SOURCES}")
message(STATUS "Header files: ${HEADERS}")
