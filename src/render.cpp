#include "render.h"
#include "common.h"
#include "simulation.h"

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

bool Renderer::init()
{
	// Create shaders
	char const* vertexShaderSource = R"(
#version 330 core
#extension GL_ARB_explicit_uniform_location : require
#extension GL_ARB_explicit_attrib_location : require
#extension GL_ARB_separate_shader_objects : require

layout (location = 0) in vec3 inPosition;
layout (location = 1) in vec3 inNormal;
// layout (location = 2) in vec3 inModelPosition;

layout (location = 0) uniform vec3 uniModelPosition;
layout (location = 1) uniform mat4 uniProj;
layout (location = 2) uniform mat4 uniView;

layout (location = 0) out vec3 vertNormal;

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
	// gl_Position = uniProj * uniView * uniModel * vec4(inPosition, 1.0);
	mat4 model = makeModelMatrix(0.02, uniModelPosition);
	gl_Position = uniProj * uniView * model * vec4(inPosition, 1.0);
}
	)";
	uint32_t vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
	glCompileShader(vertexShader);
	if (!gl_check_shader_compilation(vertexShader)) {
		logError("Vertex shader could not be created!");
		return false;
	}

	char const* fragmentShaderSource = R"(
#version 330 core
#extension GL_ARB_explicit_attrib_location : require
#extension GL_ARB_separate_shader_objects : require

layout (location = 0) in vec3 vertNormal;

layout (location = 0) out vec4 fragColor;

const vec3 lightDir = vec3(1, 1, 0.5);
const vec3 albedo = vec3(0.23, 0.42, 0.84);
const vec3 ambient = 0.3 * vec3(1.0, 0.8, 0.6);

void main()
{
	float diffuse = dot(normalize(vertNormal), normalize(lightDir));
	vec3 color = albedo * diffuse + ambient;
	fragColor = vec4(color, 1.0);
}
	)";
	uint32_t fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
	glCompileShader(fragmentShader);
	if (!gl_check_shader_compilation(fragmentShader)) {
		logError("Fragment shader could not be created!");
		return false;
	}

	shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);
	if (!gl_check_shader_linking(shaderProgram)) {
		return false;
	}

	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	glUseProgram(shaderProgram);

	// Create mesh
	shapes::Sphere sphere {};
	Mesh sphereMesh = shapes::createMesh(sphere);
	sphereTrisCount = sphereMesh.indices.size();

	// Create vertex array
	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);

	// Create vertex buffer
	glGenBuffers(1, &vertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex) * sphereMesh.vertices.size(), &sphereMesh.vertices[0], GL_STATIC_DRAW);

	// Create index buffer
	glGenBuffers(1, &indexBuffer);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(uint32_t) * sphereMesh.indices.size(), &sphereMesh.indices[0], GL_STATIC_DRAW);

	// Create instance buffer
	glGenBuffers(1, &instanceBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, instanceBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions, GL_DYNAMIC_DRAW);

	// Define vertex attributes
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, position));
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));
	glEnableVertexAttribArray(1);

	return true;
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
	glUseProgram(shaderProgram);
	glBindVertexArray(VAO);

	glm::mat4 proj = glm::perspective(glm::radians(45.f), 1.f, 0.01f, 100.f);
	glUniformMatrix4fv(1, 1, GL_FALSE, glm::value_ptr(proj));

	glm::mat4 view = glm::lookAt(scene.cameraPosition, glm::vec3{ 0 }, glm::vec3{ 0, 1, 0 });
	glUniformMatrix4fv(2, 1, GL_FALSE, glm::value_ptr(view));

	// logDebug("===== New Frame =====");
	for (int i = 0; i < scene.count; i++) {
		// glm::mat4 model(1.f);
		// model = glm::translate(model, scene.positions[i]);
		// model = glm::scale(model, glm::vec3{ 0.02f });
		// glUniformMatrix4fv(0, 1, GL_FALSE, glm::value_ptr(model));
		glUniform3fv(0, 1, glm::value_ptr(scene.positions[i]));

		glDrawElements(GL_TRIANGLES, sphereTrisCount, GL_UNSIGNED_INT, 0);
		// logDebug("Drawing sphere %d", i);
	}
}
