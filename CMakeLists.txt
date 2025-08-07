cmake_minimum_required(VERSION 3.16)
project(mppi_cuda LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

# Compile flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)

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