#include "render.h"
#include "common.h"
#include "simulation.h"

#include <glad/glad.h>

#include "config.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

bool gl_check_shader_compilation(uint32_t shader)
{
	int success;
	char infoLog[512];
	glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
	if (!success) {
		glGetShaderInfoLog(shader, 512, nullptr, infoLog);
		logError("Shader compilation error:\n%s", infoLog);
	}
	return success;
}

bool gl_check_shader_linking(uint32_t program)
{
	int success;
	char infoLog[512];
	glGetProgramiv(program, GL_LINK_STATUS, &success);
	if (!success) {
		glGetProgramInfoLog(program, 512, nullptr, infoLog);
		logError("Shader linking error:\n%s", infoLog);
	}
	return success;
}

Result Renderer::init()
{
	// Create shaders
	char const* vertexShaderSource = R"(
#version 450 core

layout (location = 0) in vec3  inPosition;
layout (location = 1) in vec3  inNormal;
layout (location = 2) in vec3  inInstancePosition;
layout (location = 3) in vec3  inInstanceVelocity;
layout (location = 4) in float inInstanceDensity;
layout (location = 5) in float inInstancePressure;
layout (location = 6) in vec3  inInstanceAcceleration;

layout (location = 0) uniform float uniScale;
layout (location = 1) uniform mat4  uniProj;
layout (location = 2) uniform mat4  uniView;

layout (location = 0)      out vec3  vertNormal;
layout (location = 1) flat out vec3  vertVelocity;
layout (location = 2) flat out float vertDensity;
layout (location = 3) flat out float vertPressure;
layout (location = 4) flat out vec3  vertAcceleration;

mat4 makeModelMatrix(in float scale, in vec3 translation) {
	mat4 model = mat4(
		vec4(scale,     0,     0, 0),
		vec4(    0, scale,     0, 0),
		vec4(    0,     0, scale, 0),
		vec4(        translation, 1)
	);
	return model;
}

void main()
{
	vertNormal = inNormal;
	mat4 model = makeModelMatrix(uniScale, inInstancePosition);
	gl_Position = uniProj * uniView * model * vec4(inPosition, 1.0);

	vertVelocity     = inInstanceVelocity;
	vertDensity      = inInstanceDensity;
	vertPressure     = inInstancePressure;
	vertAcceleration = inInstanceAcceleration;
}
	)";
	uint32_t vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
	glCompileShader(vertexShader);
	if (!gl_check_shader_compilation(vertexShader)) {
		logError("Vertex shader could not be created!");
		return FLSIM_ERROR;
	}

	char const* fragmentShaderSource = R"(
#version 450 core

layout (location = 0)      in vec3  vertNormal;
layout (location = 1) flat in vec3  vertVelocity;
layout (location = 2) flat in float vertDensity;
layout (location = 3) flat in float vertPressure;
layout (location = 4) flat in vec3  vertAcceleration;

layout (location = 0) out vec4 fragColor;

layout (location = 3) uniform int uniDisplayMode;

const vec3 lightDir = vec3(1, 1, 0.5);
const vec3 albedo = vec3(0.23, 0.42, 0.84);
const vec3 ambient = 0.3 * vec3(1.0, 0.8, 0.6);

void main()
{
	switch (uniDisplayMode) {
		case 0: // Particles
		{
			float diffuse = dot(normalize(vertNormal), normalize(lightDir));
			vec3 color = albedo * diffuse + ambient;
			fragColor = vec4(color, 1.0);
			break;
		}
		case 1: // Density
		{
			float level = vertDensity / 2.0;
			vec3 color;
			color.r = sin(level);
			color.g = sin(level * 2.0);
			color.b = cos(level);
			fragColor = vec4(color, 1.0);
			break;
		}
		case 2: // Velocity
		{
			float level = length(vertVelocity) / 15.f;
			vec3 color;
			color.r = sin(level);
			color.g = sin(level * 2.0);
			color.b = cos(level);
			fragColor = vec4(color, 1.0);
			break;
		}
		case 3: // Pressure
		{
			float level = abs(vertPressure) / 20.0;
			vec3 color;
			color.r = sin(level);
			color.g = sin(level * 2.0);
			color.b = cos(level);
			fragColor = vec4(color, 1.0);
			break;
		}
		case 4: // Acceleration
		{
			float level = length(vertAcceleration) / 10.0;
			vec3 color;
			color.r = sin(level);
			color.g = sin(level * 2.0);
			color.b = cos(level);
			fragColor = vec4(color, 1.0);
			break;
		}
	}

}
	)";
	uint32_t fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
	glCompileShader(fragmentShader);
	if (!gl_check_shader_compilation(fragmentShader)) {
		logError("Fragment shader could not be created!");
		return FLSIM_ERROR;
	}

	shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);
	if (!gl_check_shader_linking(shaderProgram)) {
		return FLSIM_ERROR;
	}

	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	glUseProgram(shaderProgram);

	// Create mesh
	shapes::Sphere sphere {};
	Mesh sphereMesh = shapes::createMesh(sphere);
	sphereTrisCount = sphereMesh.indices.size();

	// Create vertex array
	glCreateVertexArrays(1, &VAO);

	// Create vertex buffer
	glCreateBuffers(1, &vertexBuffer);
	glNamedBufferStorage(vertexBuffer, sizeof(Vertex) * sphereMesh.vertices.size(), &sphereMesh.vertices[0], 0);
	glVertexArrayVertexBuffer(VAO, 0, vertexBuffer, 0, sizeof(Vertex));
	glVertexArrayBindingDivisor(VAO, 0, 0); // Step per vertex
	glObjectLabel(GL_BUFFER, vertexBuffer, -1, "Sphere Vertices");

	// Create index buffer
	glCreateBuffers(1, &indexBuffer);
	glNamedBufferStorage(indexBuffer, sizeof(uint32_t) * sphereMesh.indices.size(), &sphereMesh.indices[0], 0);
	glVertexArrayElementBuffer(VAO, indexBuffer);
	glObjectLabel(GL_BUFFER, indexBuffer, -1, "Sphere Indices");

	// Create instance buffer
	glCreateBuffers(1, &instanceBuffer);
	glNamedBufferStorage(instanceBuffer, (3 * sizeof(float3) + 2 * sizeof(float)) * gSimCfg.NumParticles, nullptr, GL_DYNAMIC_STORAGE_BIT);
	glVertexArrayVertexBuffer(VAO, 1, instanceBuffer, 0, sizeof(float3)); // position
	glVertexArrayBindingDivisor(VAO, 1, 1); // Step per instance
	glVertexArrayVertexBuffer(VAO, 2, instanceBuffer, sizeof(float3) * gSimCfg.NumParticles, sizeof(float3)); // velocity
	glVertexArrayBindingDivisor(VAO, 2, 1); // Step per instance
	glVertexArrayVertexBuffer(VAO, 3, instanceBuffer, (2 * sizeof(float3)) * gSimCfg.NumParticles, sizeof(float)); // density
	glVertexArrayBindingDivisor(VAO, 3, 1); // Step per instance
	glVertexArrayVertexBuffer(VAO, 4, instanceBuffer, (2 * sizeof(float3) + sizeof(float)) * gSimCfg.NumParticles, sizeof(float)); // pressure
	glVertexArrayBindingDivisor(VAO, 4, 1); // Step per instance
	glVertexArrayVertexBuffer(VAO, 5, instanceBuffer, (2 * sizeof(float3) + 2 * sizeof(float)) * gSimCfg.NumParticles, sizeof(float)); // acceleration
	glVertexArrayBindingDivisor(VAO, 5, 1); // Step per instance
	glObjectLabel(GL_BUFFER, instanceBuffer, -1, "Particle Properties");

	// Define vertex attributes
	// - position
	glEnableVertexArrayAttrib(VAO, 0);
	glVertexArrayAttribBinding(VAO, 0, 0); // Bind to vertex buffer binding 0
	glVertexArrayAttribFormat(VAO, 0, 3, GL_FLOAT, GL_FALSE, offsetof(Vertex, position));
	// - normal
	glEnableVertexArrayAttrib(VAO, 1);
	glVertexArrayAttribBinding(VAO, 1, 0); // Bind to vertex buffer binding 0
	glVertexArrayAttribFormat(VAO, 1, 3, GL_FLOAT, GL_FALSE, offsetof(Vertex, normal));
	// - instance position
	glEnableVertexArrayAttrib(VAO, 2);
	glVertexArrayAttribBinding(VAO, 2, 1); // Bind to vertex buffer binding 1
	glVertexArrayAttribFormat(VAO, 2, 3, GL_FLOAT, GL_FALSE, 0);
	// - instance velocity
	glEnableVertexArrayAttrib(VAO, 3);
	glVertexArrayAttribBinding(VAO, 3, 2); // Bind to vertex buffer binding 2
	glVertexArrayAttribFormat(VAO, 3, 3, GL_FLOAT, GL_FALSE, 0);
	// - instance density
	glEnableVertexArrayAttrib(VAO, 4);
	glVertexArrayAttribBinding(VAO, 4, 3); // Bind to vertex buffer binding 3
	glVertexArrayAttribFormat(VAO, 4, 1, GL_FLOAT, GL_FALSE, 0);
	// - instance pressure
	glEnableVertexArrayAttrib(VAO, 5);
	glVertexArrayAttribBinding(VAO, 5, 4); // Bind to vertex buffer binding 4
	glVertexArrayAttribFormat(VAO, 5, 1, GL_FLOAT, GL_FALSE, 0);
	// - instance acceleration
	glEnableVertexArrayAttrib(VAO, 6);
	glVertexArrayAttribBinding(VAO, 6, 5); // Bind to vertex buffer binding 5
	glVertexArrayAttribFormat(VAO, 6, 3, GL_FLOAT, GL_FALSE, 0);


		char const* lineVertexShaderSource = R"(
#version 450 core

layout (location = 0) uniform vec3  uniScale;
layout (location = 1) uniform mat4  uniProj;
layout (location = 2) uniform mat4  uniView;

mat4 makeModelMatrix(in vec3 scale) {
	mat4 model = mat4(
		vec4(scale.x,       0,       0,   0),
		vec4(      0, scale.y,       0,   0),
		vec4(      0,       0, scale.z,   0),
		vec4(      0,       0,       0,   1)
	);
	return model;
}

vec3 positions[8] = vec3[](
	vec3(-0.5, -0.5, -0.5),
	vec3( 0.5, -0.5, -0.5),
	vec3( 0.5,  0.5, -0.5),
	vec3(-0.5,  0.5, -0.5),

	vec3(-0.5, -0.5,  0.5),
	vec3( 0.5, -0.5,  0.5),
	vec3( 0.5,  0.5,  0.5),
	vec3(-0.5,  0.5,  0.5)
);

uint indices[24] = uint[](
	0, 1, 1, 2, 2, 3, 3, 0,
	4, 5, 5, 6, 6, 7, 7, 4,
	0, 4, 1, 5, 2, 6, 3, 7
);

void main()
{
	uint index = indices[gl_VertexID];
	vec3 pos = positions[index];
	mat4 model = makeModelMatrix(uniScale);
	gl_Position = uniProj * uniView * model * vec4(pos, 1.0);
}
	)";
	uint32_t lineVertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(lineVertexShader, 1, &lineVertexShaderSource, nullptr);
	glCompileShader(lineVertexShader);
	if (!gl_check_shader_compilation(lineVertexShader)) {
		logError("Line Vertex shader could not be created!");
		return FLSIM_ERROR;
	}

	char const* lineFragmentShaderSource = R"(
#version 450 core

layout (location = 0) out vec4 fragColor;

void main()
{
	fragColor = vec4(1, 1, 1, 1);
}
	)";
	uint32_t lineFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(lineFragmentShader, 1, &lineFragmentShaderSource, nullptr);
	glCompileShader(lineFragmentShader);
	if (!gl_check_shader_compilation(lineFragmentShader)) {
		logError("Line Fragment shader could not be created!");
		return FLSIM_ERROR;
	}

	lineShaderProgram = glCreateProgram();
	glAttachShader(lineShaderProgram, lineVertexShader);
	glAttachShader(lineShaderProgram, lineFragmentShader);
	glLinkProgram(lineShaderProgram);
	if (!gl_check_shader_linking(lineShaderProgram)) {
		return FLSIM_ERROR;
	}

	glDeleteShader(lineVertexShader);
	glDeleteShader(lineFragmentShader);

	return FLSIM_SUCCESS;
}

void Renderer::destroy()
{
	glBindVertexArray(0);
	glUseProgram(0);

	glDeleteBuffers(1, &vertexBuffer);
	glDeleteBuffers(1, &indexBuffer);
	glDeleteBuffers(1, &instanceBuffer);
	glDeleteVertexArrays(1, &VAO);
	glDeleteProgram(shaderProgram);
}

void Renderer::draw(Scene &scene)
{
	// logDebug("===== New Frame =====");
	glUseProgram(shaderProgram);
	glBindVertexArray(VAO);

	float scale = scene.particleSize;
	glUniform1f(0, scale);

	glm::mat4 proj = glm::perspective(glm::radians(45.f), 1.f, 0.01f, 100.f);
	glUniformMatrix4fv(1, 1, GL_FALSE, glm::value_ptr(proj));

	glm::mat4 view = glm::lookAt(scene.cameraPosition, glm::vec3{ 0 }, glm::vec3{ 0, 1, 0 });
	glUniformMatrix4fv(2, 1, GL_FALSE, glm::value_ptr(view));

	int displayMode = scene.displayMode;
	glUniform1i(3, displayMode);

	int numDrawCalls = gSimCfg.NumParticles / FLSIM_GL_MAX_INSTANCES;
	for (int i = 0; i < numDrawCalls; i++)
	{
		glDrawElementsInstanced(GL_TRIANGLES, (int)sphereTrisCount, GL_UNSIGNED_INT, nullptr, gSimCfg.NumParticles);
	}
	glDrawElementsInstanced(GL_TRIANGLES, (int)sphereTrisCount, GL_UNSIGNED_INT, nullptr, gSimCfg.NumParticles % FLSIM_GL_MAX_INSTANCES); // Draw remaining particles


	glUseProgram(lineShaderProgram);
	glm::vec3 region = scene.region;
	glUniform3fv(0, 1, glm::value_ptr(region));

	glUniformMatrix4fv(1, 1, GL_FALSE, glm::value_ptr(proj));
	glUniformMatrix4fv(2, 1, GL_FALSE, glm::value_ptr(view));

	glDrawArrays(GL_LINES, 0, 24);
}
