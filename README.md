## Overview

Implementación del método SPH para simulación de fluidos mediante el uso de C++ y CUDA. La visualización de la simulación se lleva a cabo a través de OpenGL, utilizando la interoperabilidad CUDA/OpenGL.

## Build

El proyecto se basa en CMake, una herramienta de software libre que facilita el proceso de generación de makefiles para la compilación de código en diversos sistemas operativos y entornos de desarrollo. Para compilar el proyecto, se recomienda hacer uso de CMake para generar los makefiles correspondientes.

El código ha sido probado en dos configuraciones de sistema operativo diferentes:

 - Windows 10 de 64 bits, CMake 3.26.2 y MSVC 19.35.32217.1.
 - Ubuntu 22.04 de 64 bits, CMake 0.0.0 y GCC 11.3.0.

En ambos casos se ha testeado con CUDA 12.1.

## Instrucciones para construir el proyecto

    1. !mkdir build && cd build
    2. !cmake --build . --config Release
    3. ./Release/SPH(.exe en Windows)

Nota: El paso 3 debe realizarse desde la carpeta build ya que las rutas a recursos son relativas a esta carpeta.