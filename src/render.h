#pragma once
#include "shapes.h"

#include <glad/glad.h>

struct Scene
{
	glm::vec3 cameraPosition;

	size_t count;
	glm::vec3 *positions;
};

struct Renderer
{
	bool init();
	void destroy();

	void draw(Scene &scene);
	
	uint32_t vertexBuffer;
	uint32_t indexBuffer;
	uint32_t instanceBuffer;
	uint32_t VAO;
	uint32_t shaderProgram;

	size_t sphereTrisCount;
};
