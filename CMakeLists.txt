cmake_minimum_required(VERSION 3.10)
project(PTMS_Project)

set(CMAKE_CXX_STANDARD 17)

# Настройки для Windows (MinGW)
if(WIN32)
    add_definitions(-DGLEW_STATIC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static")
endif()

# Подключаем директории GLFW и GLEW
include_directories(
    ${CMAKE_SOURCE_DIR}/libs/glfw/include
    # ${CMAKE_SOURCE_DIR}/libs/glew/include
    ${CMAKE_SOURCE_DIR}/include  # Добавляем нашу директорию с заголовками
)

# Создаем исполняемый файл
add_executable(${PROJECT_NAME}
    src/main.cpp
    src/random_variable.cpp
    src/visualizer.cpp
)

# Линковка для Windows (MinGW)
if(WIN32)
    target_link_libraries(${PROJECT_NAME}
        ${CMAKE_SOURCE_DIR}/libs/glfw/lib/libglfw3.a
        opengl32
        gdi32
    )
else()
    # Для Linux
    target_link_libraries(${PROJECT_NAME} glfw GLEW GL)
endif()