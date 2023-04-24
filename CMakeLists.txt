
# target_compile_options(${CMAKE_PROJECT_NAME}
#     PRIVATE $<$<COMPILE_LANGUAGE:CXX: >
#     PRIVATE $<$<COMPILE_LANGUAGE:CUDA: >
# )

cmake_minimum_required(VERSION 3.5)
message(STATUS "CMake version: ${CMAKE_VERSION}")

project(SPH)

option(USE_CUDA "Use CUDA" ON)

set(CMAKE_CXX_STANDARD 20)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/extern/glm/inc
    #${CMAKE_SOURCE_DIR}/extern/glad/include/glad
    ${CMAKE_SOURCE_DIR}/extern/glfw/include
    ${CMAKE_SOURCE_DIR}/extern/glfw/deps
)

# Source files to compile
file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cu")

if( ("${CMAKE_VERSION}" VERSION_EQUAL 3.8) OR ("${CMAKE_VERSION}" VERSION_GREATER 3.8) )
    # Modern CMake
    if(USE_CUDA)
        enable_language("CUDA")
        add_executable(SPH ${cpu_source_files} ${gpu_source_files})
    else(USE_CUDA)
        add_executable(SPH ${cpu_source_files})
    endif()
else()
    # Old CMake
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

add_subdirectory(${CMAKE_SOURCE_DIR}/extern/glfw)
#find_package(OpenGL REQUIRED)

target_link_options(SPH PRIVATE "/NODEFAULTLIB:MSVCRT")
target_link_options(SPH PRIVATE "/NODEFAULTLIB:LIBCMT")
#target_link_libraries(SPH ${SPH_SOURCE_DIR}/extern/glfw/lib/glfw3.lib glad)
target_link_libraries(SPH glfw)