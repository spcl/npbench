cmake_minimum_required(VERSION 3.16)
project(blur)

# Set up language settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# Find Halide
find_package(Halide REQUIRED)
find_package(OpenMP)

# Generator
add_halide_generator(blur.generator SOURCES blur_generator.cpp)
add_halide_library(blur_auto_schedule FROM blur.generator
                   GENERATOR blur
                   AUTOSCHEDULER Halide::Mullapudi2016)

# Filters
add_halide_library(blur FROM blur.generator)

# Main executable
add_executable(blur_process process.cpp)
target_link_libraries(blur_process
                      PRIVATE
                      Halide::ImageIO
                      blur
                      blur_auto_schedule)
