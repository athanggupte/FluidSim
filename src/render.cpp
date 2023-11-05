#include "render.h"
#include "common.h"

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

bool Scene::init(SceneDesc desc)
{
	sceneDesc = desc;

	// Create shaders
	char const* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 inPosition;
layout (location = 1) in vec3 inNormal;

out vec3 vertNormal;

void main()
{
	vertNormal = inNormal;
	gl_Position = vec4(inPosition, 1.0);
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
in vec3 vertNormal;

out vec4 fragColor;

const vec3 lightDir = vec3(1, 1, 0.5);
const vec3 albedo = vec3(0.23, 0.42, 0.84);

void main()
{
	float diffuse = dot(normalize(vertNormal), normalize(lightDir));
	fragColor = vec4(albedo * diffuse, 1.0);
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
	Mesh sphereMesh = shapes::createMesh(desc.sphere);
	sphereTrisCount = sphereMesh.indices.size();

	// Create vertex array
	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);

	// Create vertex buffer
	glGenBuffers(1, &vertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex) * sphereMesh.vertices.size(), &sphereMesh.vertices[0], GL_STATIC_DRAW);

	// Craete index buffer
	glGenBuffers(1, &indexBuffer);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(uint32_t) * sphereMesh.indices.size(), &sphereMesh.indices[0], GL_STATIC_DRAW);

	// Define vertex attributes
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, position));
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));
	glEnableVertexAttribArray(1);

	return true;
}

void Scene::destroy()
{
	glBindVertexArray(0);
	glUseProgram(0);

	glDeleteBuffers(1, &vertexBuffer);
	glDeleteBuffers(1, &indexBuffer);
	glDeleteVertexArrays(1, &VAO);
	glDeleteProgram(shaderProgram);
}

void Scene::draw()
{
	glUseProgram(shaderProgram);
	glBindVertexArray(VAO);
	glDrawElements(GL_TRIANGLES, sphereTrisCount, GL_UNSIGNED_INT, 0);
}
