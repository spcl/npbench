cmake_minimum_required(VERSION 3.16)
project(conv2d_bias)

# Set up language settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# Find Halide
find_package(Halide REQUIRED)

# Generator
add_halide_generator(conv2d_bias.generator SOURCES conv2d_bias_generator.cpp)

# Filters
add_halide_library(conv2d_bias FROM conv2d_bias.generator)
add_halide_library(conv2d_bias_auto_schedule FROM conv2d_bias.generator
                   GENERATOR conv2d_bias
                   AUTOSCHEDULER Halide::Mullapudi2016)

# Main executable
add_executable(conv2d_bias_process process.cpp)
target_link_libraries(conv2d_bias_process
                      PRIVATE
                      Halide::ImageIO
                      conv2d_bias
                      conv2d_bias_auto_schedule)
