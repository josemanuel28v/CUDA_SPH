## Overview

Implementation of the SPH method for fluid simulation. The project is written in C++ and CUDA. The simulation is visualized using OpenGL, with CUDA/OpenGL interoperability.

## Build information

The project is based on CMake, so using this tool is recommended as it simplifies the process of generating makefiles for compiling code across different operating systems and development environments.

The code has been tested on two different operating system configurations:

 - Windows 10 64-bit, CMake 3.26.2 and MSVC 19.35.32217.1 (Visual Studio 17 2022).
 - Ubuntu 22.04 64-bit, CMake 3.22.1 and GCC 11.3.0.

In both cases, it has been tested with CUDA 12.1.

Note: The GPU architecture is specified in the CMakeLists.txt file.

## Build instructions

    From the project root directory, open a terminal on Linux or PowerShell on Windows:

        1. mkdir build
        2. cd build
        3. cmake -DCMAKE_BUILD_TYPE=Release ..
        4. cmake --build . --config Release
        5. cd Release
        6. ./SPH (Linux)
        6. ./SPH.exe (Windows)

Note: Step 6 must be executed from the Release directory, as resource paths are relative to this directory.

## Controls

    - Arrow keys: Camera movement.
    - Right mouse button + mouse movement: Camera rotation
    - R key: Reset the simulation.
    - P key: Pause the simulation.
