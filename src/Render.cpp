#include "Render.h"
#include "cuda_gl_interop.h"
#include <vector>

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
    gladLoadGL(glfwGetProcAddress); // con gl.h (no se enlaza explicitamente con glad.lib)
    //gladLoadGLLoader((GLADloadproc) glfwGetProcAddress); // con glad.h (se enlaza explicitamente con glad.lib)

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
}

void Render::setupObject(Object *obj, unsigned numInstances)
{
    Mesh3D* mesh = obj->getMesh();

    if (!mesh) return;

    if (bufferObjects.find(mesh->getId()) == bufferObjects.end())
    {
        std::cout << "Setting up particle " << std::endl;
        VAO_t vao;

        std::vector<vertex_t>* vertices = mesh->getVertices();
        std::vector<glm::uint32>* indices = mesh->getIndices();
        Material* mat = mesh->getMaterial();
        RenderProgram* program = mat->getProgram();

        glGenVertexArrays(1, &vao.id);
        glBindVertexArray(vao.id);

        glGenBuffers(1, &vao.v_id);
        glGenBuffers(1, &vao.i_id);
        glGenBuffers(1, &vao.mvp_id);

        // Vertices
        glBindBuffer(GL_ARRAY_BUFFER, vao.v_id);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertex_t) * vertices->size(), vertices->data(), GL_STATIC_DRAW);

        // Indices 
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vao.i_id);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * indices->size(), indices->data(), GL_STATIC_DRAW);
        
        glEnableVertexAttribArray(program->vars["vpos"]);
        glVertexAttribPointer(program->vars["vpos"], 4, GL_FLOAT, GL_FALSE, sizeof(vertex_t), (void*)offsetof(vertex_t, position));

        if (mat->getTexturing())
        {
            glEnableVertexAttribArray(program->vars["vtextcoord"]);
            glVertexAttribPointer(program->vars["vtextcoord"], 2, GL_FLOAT, GL_FALSE, sizeof(vertex_t), (void*)offsetof(vertex_t, textCoord));
        }

        // Matrices mvps (instancing)
        // glBindBuffer(GL_ARRAY_BUFFER, vao.mvp_id);
        // glBufferData(GL_ARRAY_BUFFER, sizeof(glm::mat4) * numInstances, nullptr, GL_DYNAMIC_DRAW);

        // glEnableVertexAttribArray(program->vars["vmvp"] + 0);
        // glVertexAttribPointer(program->vars["vmvp"] + 0, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (GLvoid*)0x00);
        // glVertexAttribDivisor(program->vars["vmvp"] + 0, 1);

        // glEnableVertexAttribArray(program->vars["vmvp"] + 1);
        // glVertexAttribPointer(program->vars["vmvp"] + 1, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (GLvoid*)(1 * sizeof(float) * 4));
        // glVertexAttribDivisor(program->vars["vmvp"] + 1, 1);

        // glEnableVertexAttribArray(program->vars["vmvp"] + 2);
        // glVertexAttribPointer(program->vars["vmvp"] + 2, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (GLvoid*)(2 * sizeof(float) * 4));
        // glVertexAttribDivisor(program->vars["vmvp"] + 2, 1);

        // glEnableVertexAttribArray(program->vars["vmvp"] + 3);
        // glVertexAttribPointer(program->vars["vmvp"] + 3, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (GLvoid*)(3 * sizeof(float) * 4));
        // glVertexAttribDivisor(program->vars["vmvp"] + 3, 1);

        // Positions instead of mvps
        glBindBuffer(GL_ARRAY_BUFFER, vao.mvp_id);
        glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec4) * numInstances, nullptr, GL_DYNAMIC_DRAW);

        glEnableVertexAttribArray(program->vars["ppos"] + 0);
        glVertexAttribPointer(program->vars["ppos"], 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (GLvoid*)0x00);
        glVertexAttribDivisor(program->vars["ppos"], 1);

        glBindVertexArray(0);

        bufferObjects[mesh->getId()] = vao;

        // CUDA register (mapear posiciones en GPU OpenGL <-> CUDA)
        // cudaGLSetGLDevice(0); ?? no se si es necesario
        cudaGraphicsGLRegisterBuffer(&vao.cuda_id, vao.mvp_id, cudaGraphicsMapFlagsWriteDiscard);
    }
}

void Render::removeObject(Object *obj)
{
    Mesh3D* mesh = obj->getMesh();

    if (!mesh) return;

    uint meshId = mesh->getId();

    if (bufferObjects.find(meshId) != bufferObjects.end())
    {
        const VAO_t& vbo = bufferObjects[meshId];
        glDeleteVertexArrays(1, &vbo.id);
        glDeleteBuffers(1, &vbo.v_id);
        glDeleteBuffers(1, &vbo.i_id);
    }
}

#include <glm/gtx/string_cast.hpp>
void Render::drawObject(Object *obj, unsigned numInstances, glm::vec4* positions)
{
    Mesh3D* mesh = obj->getMesh();
    Material* mat = mesh->getMaterial();

    // Activar buffers antes de usar el programa
    VAO_t buffer = bufferObjects[mesh->getId()];

    // Attributes
    mat->prepare();

    // Dibujado
    glBindVertexArray(buffer.id);
    glBindBuffer(GL_ARRAY_BUFFER, buffer.mvp_id);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec4) * numInstances, &positions[0], GL_DYNAMIC_DRAW);
    //glBindBuffer(GL_ARRAY_BUFFER, buffer.color_id);
    //glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec4) * numInstances, &colors[0], GL_DYNAMIC_DRAW);
    glDrawElementsInstanced(GL_TRIANGLES, mesh->getIndices()->size(), GL_UNSIGNED_INT, nullptr, numInstances); 
    glBindVertexArray(0);
}

void Render::clearDisplay()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void Render::swapBuffers()
{
    glfwSwapBuffers(window);
}

bool Render::isClosed()
{
    return bool(glfwWindowShouldClose(window));
}