cmake_minimum_required(VERSION 3.16)
project(hist)

# Set up language settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# Find Halide
find_package(Halide REQUIRED)

# Generator
add_halide_generator(hist.generator SOURCES hist_generator.cpp)

# Filters
add_halide_library(hist FROM hist.generator)
add_halide_library(hist_auto_schedule FROM hist.generator
                   GENERATOR hist
                   AUTOSCHEDULER Halide::Mullapudi2016)

# Main executable
add_executable(hist_process process.cpp)
target_link_libraries(hist_process
                      PRIVATE
                      likwid::likwid
                      Halide::ImageIO
                      Halide::Tools
                      hist
                      hist_auto_schedule)
