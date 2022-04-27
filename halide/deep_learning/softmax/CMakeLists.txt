cmake_minimum_required(VERSION 3.16)
project(softmax)

# Set up language settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# Find Halide
find_package(Halide REQUIRED)

# Generator
add_halide_generator(softmax.generator SOURCES softmax_generator.cpp)

# Filters
add_halide_library(softmax FROM softmax.generator)
add_halide_library(softmax_auto_schedule FROM softmax.generator
                   GENERATOR softmax
                   AUTOSCHEDULER Halide::Mullapudi2016)

# Main executable
add_executable(softmax_process process.cpp)
target_link_libraries(softmax_process
                      PRIVATE
                      Halide::ImageIO
                      softmax
                      softmax_auto_schedule)
