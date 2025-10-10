cmake_minimum_required(VERSION 3.10)
project(DAA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(DAA
    main.cpp
    src/openmp.cpp
    src/utils.cpp
)

target_include_directories(DAA PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)