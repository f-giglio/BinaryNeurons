cmake_minimum_required(VERSION 3.20)

project(BinaryNeurons
    VERSION 1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)

add_executable(
    FixedPointFirstVersion
    ./CompositeClassic/FixedPointFirstVersion.cpp
)

target_link_libraries(
    FixedPointFirstVersion
    PRIVATE
        Eigen3::Eigen
        OpenMP::OpenMP_CXX
)

target_compile_options(
    FixedPointFirstVersion
    PRIVATE
        -Wall
        -Wpedantic
        $<$<CONFIG:Release>:-O3>
)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Compilazione in modalità Release")
endif()
