## Overview

Implementación del método SPH para simulación de fluidos. El proyecto está escrito en C++ y CUDA. La visualización de la simulación se lleva a cabo a través de OpenGL, utilizando la interoperabilidad CUDA/OpenGL.

## Build information

El proyecto está basado en CMake, por lo que se recomienda el uso de esta herramienta debido a que facilita el proceso de generación de makefiles para la compilación de código en diversos sistemas operativos y entornos de desarrollo.

El código se ha probado en dos configuraciones de sistema operativo diferentes:

 - Windows 10 de 64 bits, CMake 3.26.2 y MSVC 19.35.32217.1.
 - Ubuntu 22.04 de 64 bits, CMake 3.22.1 y GCC 11.3.0.

En ambos casos se ha testeado con CUDA 12.1.

Nota: La arquitectura de la GPU está especificada en el fichero CMakeLists.txt

## Build instructions

    En la carpeta raíz del proyecto, abrir una terminal en Linux o una powershell en Windows:

        1. mkdir build
        2. cd build
        3. cmake -DCMAKE_BUILD_TYPE=Release ..
        4. cmake --build . --config Release
        5. cd Release
        6. ./SPH (Linux)
        6. ./SPH.exe (Windows)

Nota: El paso 6 debe realizarse desde la carpeta Release, ya que las rutas a recursos son relativas a esta carpeta.

## Controls

    - Flechas: movimiento de la cámara.
    - Click derecho + movimiento del ratón: rotación de la cámara
    - Tecla R: reiniciar la simulación.
    - Tecla P: pausar la simulación.
