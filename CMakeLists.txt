cmake_minimum_required(VERSION 3.18)
project(mppi_cuda LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Compile flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Include directory
include_directories(include
                    /usr/include/eigen3
)

# Source files
set(TEST_SOURCES
    src/test.cpp
    src/mppi_solver.cu
)
add_executable(test_mppi_cuda ${TEST_SOURCES})


set(SOURCES
    src/mppi_solver.cu
    main.cpp
)
add_executable(mppi_cuda ${SOURCES})