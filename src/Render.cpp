#include "Render.h"

Render::Render(uint width, uint height)
{
    this->width = width;
    this->height = height;
    window = nullptr;
}

void Render::init()
{
	// Inicializa GLFW
	if (!glfwInit())
    {
        std::cout << "ERROR GLFWINIT\n";
    }

    // Crear la ventana
    window = glfwCreateWindow(this->width, this->height, "OpenGL 4.0", nullptr, nullptr);

    if (!window)
    {
        std::cerr << "Error: no se pudo crear la ventana de GL4R." << std::endl;
        glfwTerminate();
    }
     
    glfwMakeContextCurrent(window);
    //gladLoadGL(glfwGetProcAddress);
    gladLoadGLLoader((GLADloadproc) glfwGetProcAddress);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
}

void Render::setupObject(Object *obj, unsigned numInstances)
{

}

void Render::removeObject(Object *obj, unsigned numInstances)
{

}

void Render::drawObject(Object *obj, unsigned numInstances)
{

}

void Render::clearDisplay()
{

}

void Render::swapBuffers()
{

}