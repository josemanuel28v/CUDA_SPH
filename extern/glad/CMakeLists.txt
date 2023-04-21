cmake_minimum_required(VERSION 3.5)

project(glad)

set(GLAD_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/glad.c)

add_library(glad STATIC ${GLAD_SRC_FILES}) # Crear .lib

target_include_directories(glad PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include)
