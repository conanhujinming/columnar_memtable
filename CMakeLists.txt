# CMake minimum version
cmake_minimum_required(VERSION 3.16)

# Project name and language
project(MemtableComparison CXX)

# Set C++ standard
# C++17 is sufficient for the last main.cpp I sent.
# If you use std::barrier, you would need C++20. Let's stick to C++17 for now.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Compiler Flags (Corrected Version) ---
# These are additional flags. Optimization (-O3) will be handled by CMAKE_BUILD_TYPE.
# We define them as a proper CMake list (no quotes).
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(ADDITIONAL_CXX_FLAGS -Wall -Wextra -pthread -march=native)
endif()

# --- Dependency: mimalloc ---
option(MI_BUILD_SHARED "Build mimalloc as a shared library" OFF)
option(MI_OVERRIDE "Override malloc" ON)
add_subdirectory(mimalloc)

# --- Dependency: Google Benchmark ---
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable testing of the benchmark library")
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "Disable gtest tests for benchmark library")

include(FetchContent)
FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG        v1.8.0
)
FetchContent_MakeAvailable(googlebenchmark)

# --- Define the main benchmark/test executable ---
add_executable(memtable_benchmark main.cpp)

# --- Configure Include Directories ---
target_include_directories(memtable_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/mimalloc/include
    ${googlebenchmark_SOURCE_DIR}/include
)

# --- Configure Linking ---
find_package(Threads REQUIRED)
target_link_libraries(memtable_benchmark PRIVATE
    mimalloc-static 
    Threads::Threads
    benchmark::benchmark
)

# Apply our *additional* compiler flags to the executable
target_compile_options(memtable_benchmark PRIVATE ${ADDITIONAL_CXX_FLAGS})

# --- Final Output ---
message(STATUS "Project configured to build: memtable_benchmark")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER_ID}")