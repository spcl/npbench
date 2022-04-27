cmake_minimum_required(VERSION 3.16)
project(mlp)

# Set up language settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# Find Halide
find_package(Halide REQUIRED)

# Generator
add_halide_generator(mlp.generator SOURCES mlp_generator.cpp)

# Filters
add_halide_library(mlp FROM mlp.generator)
add_halide_library(mlp_auto_schedule FROM mlp.generator
                   GENERATOR mlp
                   AUTOSCHEDULER Halide::Mullapudi2016)

# Main executable
add_executable(mlp_process process.cpp)
target_link_libraries(mlp_process
                      PRIVATE
                      Halide::ImageIO
                      mlp
                      mlp_auto_schedule)
