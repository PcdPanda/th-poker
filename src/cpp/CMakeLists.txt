cmake_minimum_required(VERSION 3.14)
set(PROJECT_NAME "pytemplate")
project(${PROJECT_NAME})
project(MyProject LANGUAGES CXX)

# Specify C++ standard (C++20 in this case) and disable extensions
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/../python/${PROJECT_NAME}")
set(CONDA_PREFIX $ENV{CONDA_PREFIX})


# Include directory (target to be used in both build and install)
include_directories(include)

# Specify the source files (recursively find all .cpp files in src)
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS src/*.cpp)

add_library(example SHARED src/example.cpp)
set_target_properties(example PROPERTIES PREFIX "")
set(shared_libraries example)
install(TARGETS ${shared_libraries})
