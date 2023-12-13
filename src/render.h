#pragma once
#include <glm/vec3.hpp>

#include "common.h"

enum DisplayMode
{
	DM_Particles,
	DM_Density,
	DM_Velocity,
	DM_Pressure,
	DM_Acceleration,
	DM_Surface,

	DM_COUNT
};

constexpr const char* DisplayModeStr[DM_COUNT] = {
	"Particles",
	"Density",
	"Velocity",
	"Pressure",
	"Acceleration",
	"Surface",
};

struct Scene
{
	glm::vec3 cameraPosition;
	float particleSize;
	glm::vec3 region;

	DisplayMode displayMode;
	float densityRange;
	float velocityRange;
	float pressureRange;
	float accelerationRange;
};

#define FLSIM_GL_MAX_INSTANCES 100000u

struct Renderer
{
	Result init();
	void destroy();

	void draw(Scene &scene);
	
	uint32_t vertexBuffer;
	uint32_t indexBuffer;
	uint32_t instanceBuffer;
	uint32_t VAO;
	uint32_t shaderProgram;
	size_t sphereTrisCount;

	uint32_t lineShaderProgram;
};
