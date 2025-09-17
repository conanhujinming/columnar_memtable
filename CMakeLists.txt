# CMake minimum version
cmake_minimum_required(VERSION 3.16)

# Project name
project(ColumnarMemtableProject CXX)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Add and configure mimalloc ---
# Add mimalloc as a subdirectory. CMake will find its CMakeLists.txt and process it.
# We set MI_BUILD_SHARED to OFF to build a static library, which is easier to link.
# MI_OVERRIDE=OFF means mimalloc won't automatically override malloc at compile time.
# We will do it by including the header in our main.cpp.
option(MI_BUILD_SHARED "Build mimalloc as a shared library" OFF)
option(MI_OVERRIDE "Override malloc" ON)
add_subdirectory(mimalloc)

# --- Define our main executable ---
# This will be our test runner or example program.
add_executable(columnar_memtable_example main.cpp)

# --- Include directories ---
# Add the main mimalloc include path for our executable
target_include_directories(columnar_memtable_example PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/mimalloc/include
)

# --- Find required packages ---
# Find the standard threads package
find_package(Threads REQUIRED)

# --- Link libraries ---
# Link our executable against the mimalloc library and pthreads
target_link_libraries(columnar_memtable_example PRIVATE
    mimalloc-static 
    Threads::Threads
)

# Optional: Add compiler flags for optimization
target_compile_options(columnar_memtable_example PRIVATE -O2 -Wall)

# Print a message to confirm the executable is being built
message(STATUS "Configured to build executable: columnar_memtable_example")