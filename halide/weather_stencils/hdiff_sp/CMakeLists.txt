cmake_minimum_required(VERSION 3.16)
project(hdiff_sp)

# Set up language settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# Find Halide
find_package(Halide REQUIRED)

# Generator
add_halide_generator(hdiff_sp.generator SOURCES hdiff_sp_generator.cpp)
add_halide_library(hdiff_sp_auto_schedule FROM hdiff_sp.generator
                   GENERATOR hdiff_sp
                   AUTOSCHEDULER Halide::Mullapudi2016)

# Filters
add_halide_library(hdiff_sp FROM hdiff_sp.generator)

# Main executable
add_executable(hdiff_sp_process process.cpp)
target_link_libraries(hdiff_sp_process
                      PRIVATE
                      likwid::likwid
		      Halide::ImageIO
                      hdiff_sp
                      hdiff_sp_auto_schedule)
