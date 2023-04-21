
# target_compile_options(${CMAKE_PROJECT_NAME}
#     PRIVATE $<$<COMPILE_LANGUAGE:CXX: >
#     PRIVATE $<$<COMPILE_LANGUAGE:CUDA: >
# )

cmake_minimum_required(VERSION 3.5)
message(STATUS "CMake version: ${CMAKE_VERSION}")

project(SPH)

option(USE_CUDA "Use CUDA" ON)

file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cu")

if( ("${CMAKE_VERSION}" VERSION_EQUAL 3.8) OR ("${CMAKE_VERSION}" VERSION_GREATER 3.8) )
    # Modern CMake
    message(STATUS "Build with modern CMake")
    if(USE_CUDA)
        enable_language("CUDA")
        add_executable(SPH ${cpu_source_files} ${gpu_source_files})
    else(USE_CUDA)
        add_executable(SPH ${cpu_source_files})
    endif()
else()
    # Old CMake
    message(STATUS "Build with old CMake")
    add_executable(SPH ${cpu_source_files})
    if(USE_CUDA)
        find_package(CUDA 8.0)
        if(NOT CUDA_FOUND)
            message(STATUS "CUDA not found")
            set(USE_CUDA OFF)
        else()
            CUDA_ADD_LIBRARY(SPH_GPU ${gpu_source_files})
            target_link_libraries(SPH SPH_GPU)
        endif()
    endif()
endif()

message(STATUS "USE_CUDA: ${USE_CUDA}")
if(USE_CUDA)
    add_definitions(-DUSE_CUDA)
endif()