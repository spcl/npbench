cmake_minimum_required(VERSION 3.16)
project(iir_blur)

# Set up language settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# Find Halide
find_package(Halide REQUIRED)

# Generator
add_halide_generator(iir_blur.generator SOURCES iir_blur_generator.cpp)

# Filters
add_halide_library(iir_blur FROM iir_blur.generator)
add_halide_library(iir_blur_auto_schedule FROM iir_blur.generator
                   GENERATOR iir_blur
                   AUTOSCHEDULER Halide::Mullapudi2016)

# Main executable
add_executable(iir_blur_process process.cpp)
target_link_libraries(iir_blur_process PRIVATE
                      Halide::Tools
                      Halide::ImageIO
                      iir_blur
                      iir_blur_auto_schedule)
