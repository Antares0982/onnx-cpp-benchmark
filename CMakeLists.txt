cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(testproject LANGUAGES CXX)

add_library(common INTERFACE)

file(GLOB_RECURSE cpp_path src/*.cpp)

if (CMAKE_BUILD_TYPE MATCHES "Rel*")
    target_compile_options(common INTERFACE -O3)
else ()
endif ()

target_compile_options(common INTERFACE -Wall -pthread -Werror=return-type)

target_link_libraries(common INTERFACE onnxruntime)

target_compile_definitions(common INTERFACE USE_MEMORY_POOL)

add_subdirectory(cmake/cuda_test)
##############################################################################
set(BUILD_TARGET onnxbenchmark)
add_executable(${BUILD_TARGET} ${cpp_path})

set_target_properties(${BUILD_TARGET} PROPERTIES OUTPUT_NAME ${BUILD_TARGET})

target_link_libraries(${BUILD_TARGET} PRIVATE common)
