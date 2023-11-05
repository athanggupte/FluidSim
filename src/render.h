#pragma once
#include "shapes.h"

#include <glad/glad.h>

struct SceneDesc
{
	shapes::Sphere sphere;
	size_t sphereCount;
};

struct Scene
{
	bool init(SceneDesc desc);
	void destroy();

	void draw();
	
	uint32_t vertexBuffer;
	uint32_t indexBuffer;
	uint32_t VAO;
	uint32_t shaderProgram;

	SceneDesc sceneDesc;
	size_t sphereTrisCount;
};
