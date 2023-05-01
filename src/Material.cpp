#include "Material.h"
#include "System.h"

Material::Material()
{
	this->program = new RenderProgram();
	this->texture = nullptr;
}

void Material::prepare()
{
    program->use();

	if (getTexturing())
	{
		texture->bind(0);
		program->setTexture2D("colorText", 0);
	}

	program->setInt("texturing", (int)texturing);
	program->setMatrix("view", System::getCamera()->getView());
	program->setMatrix("proj", System::getCamera()->getProjection());
	program->setFloat("radius", System::getParticleSystem()->getPrototype()->getRadius());

	// Depth test
	//depthWrite ? glDepthMask(GL_TRUE) : glDepthMask(GL_FALSE);
    glDepthMask(GL_TRUE);

	// Culling test
	// if (culling)
	// {
	// 	glEnable(GL_CULL_FACE);
	// 	glCullFace(GL_BACK);
	// }
	// else
	// {
	 	glDisable(GL_CULL_FACE);
	// }

	// Modo de mezclado de colores
	// switch (blendMode)
	// {
	// case SOLID:
	 	glBlendFunc(GL_ONE, GL_ZERO);
	// 	break;

	// case ALPHA:
	 	//glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	// 	break;

	// case MUL:
	// 	glBlendFunc(GL_DST_COLOR, GL_ZERO);
	// 	break;

	// case ADD:
	// 	glBlendFunc(GL_ONE, GL_ONE);
	// 	break;
	// }
}

void Material::loadPrograms(std::vector<std::string> fileNames)
{
	
	for (const auto& file : fileNames) 
	{
		if (file.ends_with("vert"))
		{
			this->program->setProgram(file, RenderProgram::RenderTypes::VERTEX);
		}
		else if (file.ends_with("frag"))
		{
			this->program->setProgram(file, RenderProgram::RenderTypes::FRAGMENT);
		}
	}

	this->program->linkPrograms();
}

Material::~Material()
{
	if (texture) delete texture;
	if (program) delete program;
}