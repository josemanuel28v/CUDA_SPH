
cmake_minimum_required(VERSION 3.5)
message(STATUS "CMake version: ${CMAKE_VERSION}")

project(SPH CUDA CXX)

option(USE_CUDA "Use CUDA" ON)

set(CMAKE_CXX_STANDARD 20)

# CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 70 75)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/extern/glm/inc
    ${CMAKE_SOURCE_DIR}/extern/glfw/include
    ${CMAKE_SOURCE_DIR}/extern/glfw/deps
)

# Source files to compile
file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cu")

if(USE_CUDA)
    enable_language("CUDA")
    add_executable(SPH ${cpu_source_files} ${gpu_source_files})
else(USE_CUDA)
    add_executable(SPH ${cpu_source_files})
endif()

target_compile_options(SPH
    # -O3 -march=native -ffast-math en gcc y clang
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>:># /O2 /arch:AVX2 /fp:fast>
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -O3 --use_fast_math># --gpu-architecture=sm_75>
)

message(STATUS "USE_CUDA: ${USE_CUDA}")
if(USE_CUDA)
    add_definitions(-DUSE_CUDA)
endif()

add_subdirectory(${CMAKE_SOURCE_DIR}/extern/glfw)

target_link_options(SPH PRIVATE "/NODEFAULTLIB:LIBCMT") # Por warnings de linkado
target_link_libraries(SPH glfw)
