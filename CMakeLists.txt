cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(testproject LANGUAGES CXX)

add_library(common INTERFACE)

file(GLOB cpp_path src/*.cpp)

target_compile_options(common INTERFACE -Wall -pthread -Werror=return-type)

target_link_libraries(common INTERFACE onnxruntime)

target_compile_definitions(common INTERFACE USE_MEMORY_POOL)

##############################################################################
set(BUILD_TARGET onnxbenchmark)
add_executable(${BUILD_TARGET} ${cpp_path})

set_target_properties(${BUILD_TARGET} PROPERTIES OUTPUT_NAME ${BUILD_TARGET})

target_link_libraries(${BUILD_TARGET} PRIVATE common)
