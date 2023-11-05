#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <glm/gtc/matrix_transform.hpp>

#include "common.h"
#include "render.h"
#include "simulation.h"

void glfw_error_callback(int error, char const* description);
void glfw_mouse_button_callback(GLFWwindow *window, int button, int action, int mods);
void glfw_cursor_pos_callback(GLFWwindow *window, double xpos, double ypos);

Scene scene;

struct WindowData
{
	float currentFrame;
	float lastFrame;
	float deltaTime;
	float fixedTotalTime = 0.f;
	float fixedDeltaTime = 0.01f;

	bool      mouseDragging { false };
	glm::vec2 mouseMovement { 0 };
	glm::vec2 prevMousePosition;
} windowData;


int main(int argc, char **argv)
{
	GLFWwindow *window;

	// Initialize CUDA
	cudaDeviceReset();
	int cuDevice;
	cudaDeviceProp cuDeviceProp;
	cudaGetDevice(&cuDevice);
	cudaGetDeviceProperties(&cuDeviceProp, cuDevice);
	logDebug("CUDA device: %d\n    Device Name: %s", cuDevice, cuDeviceProp.name);

	// Initialize GLFW library
	if (!glfwInit()) {
		logError("Could not initialize GLFW!");
		return -1;
	}
	logDebug("Initialized GLFW");

	// Set GLFW error callback
	glfwSetErrorCallback(glfw_error_callback);

	// Specify OpenGL version and profile
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Create a windowed mode window and OpenGL context
	window = glfwCreateWindow(600, 600, "FluidSim", nullptr, nullptr);
	if (!window) {
		logError("Could not create window!");
		glfwTerminate();
		return -1;
	}
	
	// Set window ser data pointer
	glfwSetWindowUserPointer(window, &windowData);

	// Make the OpenGL context current
	glfwMakeContextCurrent(window);

	// Initialize Glad loader
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		logError("Failed to initialize Glad!");
		glfwDestroyWindow(window);
		glfwTerminate();
		return -1;
	}
	logDebug("Initialized OpenGL\n    OpenGL Version: %s\n    GLSL Version: %s", glGetString(GL_VERSION), glGetString(GL_SHADING_LANGUAGE_VERSION));
	printf("    GLSL Vendor: %s\n    GLSL Renderer: %s\n", glGetString(GL_VENDOR), glGetString(GL_RENDERER));

	// Set GLFW input callbacks
	glfwSetMouseButtonCallback(window, glfw_mouse_button_callback);
	glfwSetCursorPosCallback(window, glfw_cursor_pos_callback);

	{
		double x, y;
		glfwGetCursorPos(window, &x, &y);
		windowData.prevMousePosition = {x, y};
	}

	// Set OpenGL global state
	glEnable(GL_DEPTH_TEST);
	// glDepthFunc(GL_LESS);
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);
	glFrontFace(GL_CCW);

	// Initialize OpenGL Scene and Renderer
	Renderer renderer {};
	if (!renderer.init()) {
		logError("Could not initialize renderer!");
		glfwDestroyWindow(window);
		glfwTerminate();
		return -1;
	}

	scene.cameraPosition = glm::vec3{ 0, 0, 5 };
	scene.count = std::size(positions);
	scene.positions = (glm::vec3*)&positions[0];

	for (int i = 0; i < scene.count; i++) {
		positions[i][0] = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f) * 2.f;
		positions[i][1] = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f) * 2.f;
		positions[i][2] = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f) * 2.f;

		// printf("%0.3f    %0.3f    %0.3f\n", scene.positions[i][0], scene.positions[i][1], scene.positions[i][2]);
	}

	// CUDA variables
	cudaGraphicsResource_t cuglResource;
	float *dev_positions;
	size_t dev_positionsSize;
	cudaError_t cuError;

	cuError = cudaGraphicsGLRegisterBuffer(&cuglResource, renderer.instanceBuffer, cudaGraphicsRegisterFlagsNone);
	if (cuError != cudaSuccess) {
		logError("cudaGraphicsGLRegisterBuffer Failed! : [%d] %s : %s", (int)cuError, cudaGetErrorName(cuError), cudaGetErrorString(cuError));
		renderer.destroy();
		glfwDestroyWindow(window);
		glfwTerminate();
		return -1;
	}

	// Main loop
	while (!glfwWindowShouldClose(window)) {
		// Update globals
		windowData.currentFrame = glfwGetTime();
		windowData.deltaTime = windowData.currentFrame - windowData.lastFrame;
		windowData.lastFrame = windowData.currentFrame;
		windowData.fixedTotalTime += windowData.fixedDeltaTime;

		// Update inputs
		if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
			auto cameraDirection = glm::normalize(-scene.cameraPosition);
			scene.cameraPosition += cameraDirection * (0.5f * windowData.deltaTime);
		}
		else if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
			auto cameraDirection = glm::normalize(-scene.cameraPosition);
			scene.cameraPosition -= cameraDirection * (0.5f * windowData.deltaTime);
		}

		// Update simulation
		cuError = cudaGraphicsMapResources(1, &cuglResource);
		if (cuError != cudaSuccess) {
			logError("cudaGraphicsMapResources Failed!");
			break;
		}
		cuError = cudaGraphicsResourceGetMappedPointer((void**)&dev_positions, &dev_positionsSize, cuglResource);
		if (cuError != cudaSuccess) {
			logError("cudaGraphicsResourceGetMappedPointer Failed!");
			break;
		}
		logDebug("GL buffer mapped to CUDA");

		simulateParticles(dev_positions, scene.count, windowData.fixedDeltaTime, windowData.fixedTotalTime);
		logDebug("Simulation step compelte");

		cudaGraphicsUnmapResources(1, &cuglResource);
		logDebug("GL buffer unmapped from CUDA");

		// Clear framebuffer
		glClearColor(0.6, 0.48, 0.36, 1.f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Render scene
		renderer.draw(scene);

		// Display new frame
		glfwSwapBuffers(window);

		// Check for events
		glfwPollEvents();

		break;
	}

	renderer.destroy();

	// Terminate GLFW
	glfwDestroyWindow(window);
	glfwTerminate();

	return 0;
}

void glfw_error_callback(int error, char const *description)
{
	logError("GLFW [%d]: %s", error, description);
}

void glfw_mouse_button_callback(GLFWwindow *window, int button, int action, int mods)
{
	if (button == GLFW_MOUSE_BUTTON_LEFT) {
		windowData.mouseDragging = (action == GLFW_PRESS);
	}
}

void glfw_cursor_pos_callback(GLFWwindow *window, double xpos, double ypos)
{
	glm::vec2 currMousePosition { xpos, ypos };
	windowData.mouseMovement = currMousePosition - windowData.prevMousePosition;
	windowData.prevMousePosition = currMousePosition;

	if (windowData.mouseDragging) {
		auto angles = windowData.mouseMovement * 5.f * windowData.fixedDeltaTime * -1.f;
		auto transform = glm::rotate(glm::mat4(1.f), angles.y, glm::vec3{ 1, 0, 0});
		transform *= glm::rotate(glm::mat4(1.f), angles.x, glm::vec3{ 0, 1, 0});
		scene.cameraPosition = transform * glm::vec4{ scene.cameraPosition, 1 };
	}
}
