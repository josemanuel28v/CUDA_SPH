cmake_minimum_required(VERSION 3.5)

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
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}   # Necesario en Linux para cuda_gl_interop.h
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

if(CMAKE_HOST_SYSTEM_NAME MATCHES "Windows") 
    target_compile_options(SPH
        PRIVATE $<$<COMPILE_LANGUAGE:CXX>: /O2 /arch:AVX2 /fp:fast>
        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -O3 --use_fast_math>
    )
elseif(CMAKE_HOST_SYSTEM_NAME MATCHES "Linux")
    target_compile_options(SPH
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -O3 -march=native -ffast-math>
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -O3 --use_fast_math>
    )
endif()

message(STATUS "USE_CUDA: ${USE_CUDA}")
if(USE_CUDA)
    add_definitions(-DUSE_CUDA)
endif()

add_subdirectory(${CMAKE_SOURCE_DIR}/extern/glfw)

if(CMAKE_HOST_SYSTEM_NAME MATCHES "Windows") 
    target_link_options(SPH PRIVATE "/NODEFAULTLIB:LIBCMT") # Por warnings de linkado en Windows
endif()    
target_link_libraries(SPH glfw)
