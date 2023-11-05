#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "common.h"
#include "render.h"

void glfw_error_callback(int error, char const* description);

int main(int argc, char **argv)
{
	GLFWwindow *window;

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

	// Make the OpenGL context current
	glfwMakeContextCurrent(window);

	// Initialize Glad loader
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		logError("Failed to initialize Glad!");
		glfwDestroyWindow(window);
		glfwTerminate();
		return -1;
	}

	logDebug("Initialized OpenGL Version: %s", glGetString(GL_VERSION));

	// Set OpenGL global state
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);
	glFrontFace(GL_CCW);

	SceneDesc sceneDesc {};
	Scene scene {};
	if (!scene.init(sceneDesc)) {
		logError("Could not initialize scene!");
		glfwDestroyWindow(window);
		glfwTerminate();
		return -1;
	}

	// Main loop
	while (!glfwWindowShouldClose(window)) {
		// Clear framebuffer
		glClearColor(0.f, 0.f, 0.f, 0.f);
		glClear(GL_COLOR_BUFFER_BIT);

		// Render scene
		scene.draw();

		// Display new frame
		glfwSwapBuffers(window);

		// Check for events
		glfwPollEvents();
	}

	scene.destroy();

	// Terminate GLFW
	glfwDestroyWindow(window);
	glfwTerminate();

	return 0;
}

void glfw_error_callback(int error, char const *description)
{
	logError("GLFW [%d]: %s", error, description);
}
