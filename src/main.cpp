#ifndef FLSIM_MAIN_CPP
#define FLSIM_MAIN_CPP
#endif

#include <random>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <glm/gtc/matrix_transform.hpp>

#include "common.h"
#include "config.h"
#include "render.h"
#include "simulation.h"

void glfw_error_callback(int error, char const* description);
void glfw_mouse_button_callback(GLFWwindow *window, int button, int action, int mods);
void glfw_scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void glfw_cursor_pos_callback(GLFWwindow *window, double xpos, double ypos);

void APIENTRY gl_debug_output(GLenum source, GLenum type, unsigned int id, GLenum severity, GLsizei length, const char *message, const void *userParam);

Scene gScene;
SimulationConfig gSimCfg;

struct WindowData
{
	float currentFrameTime;
	float lastFrameTime;
	float deltaTime;
	float fixedTotalTime = 0.f;
	float fixedDeltaTime = 0.001f;

	bool      mouseDragging { false };
	glm::vec2 mouseMovement { 0 };
	glm::vec2 prevMousePosition;
	glm::vec2 scrollDelta { 0 };
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
	logDebug("CUDA device properties: (Device %d)\n", cuDevice);
	printf("    Device Name           : %s\n", cuDeviceProp.name);
	printf("    Compute capability    : %d.%d\n", cuDeviceProp.major, cuDeviceProp.minor);
	printf("    # Multiprocessors     : %d\n", cuDeviceProp.multiProcessorCount);
	printf("    Max grid size         : (%d %d %d)\n", cuDeviceProp.maxGridSize[0], cuDeviceProp.maxGridSize[1], cuDeviceProp.maxGridSize[2]);
	printf("    Max block size        : (%d %d %d)\n", cuDeviceProp.maxThreadsDim[0], cuDeviceProp.maxThreadsDim[1], cuDeviceProp.maxThreadsDim[2]);
	printf("    Max threads per block : %d\n\n", cuDeviceProp.maxThreadsPerBlock);

	// Initialize GLFW library
	if (!glfwInit()) {
		logError("Could not initialize GLFW!");
		return -1;
	}
	logDebug("Initialized GLFW");

	// Set GLFW error callback
	glfwSetErrorCallback(glfw_error_callback);

	// Specify OpenGL version and profile
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, true);

	// Create a windowed mode window and OpenGL context
	window = glfwCreateWindow(800, 600, "FluidSim", nullptr, nullptr);
	if (!window) {
		logError("Could not create window!");
		glfwTerminate();
		return -1;
	}
	
	// Set window ser data pointer
	glfwSetWindowUserPointer(window, &windowData);
	
	// Set GLFW input callbacks
	glfwSetMouseButtonCallback(window, glfw_mouse_button_callback);
	glfwSetScrollCallback(window, glfw_scroll_callback);
	glfwSetCursorPosCallback(window, glfw_cursor_pos_callback);

	{
		double x, y;
		glfwGetCursorPos(window, &x, &y);
		windowData.prevMousePosition = {x, y};
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
	logDebug("Initialized OpenGL\n");
	printf("    OpenGL Version : %s\n", glGetString(GL_VERSION));
	printf("    GLSL Version   : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
	printf("    GLSL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("    GLSL Renderer  : %s\n\n", glGetString(GL_RENDERER));

	// Setup OpenGL debugging
	int flags; glGetIntegerv(GL_CONTEXT_FLAGS, &flags);
	if (flags & GL_CONTEXT_FLAG_DEBUG_BIT) {
		glEnable(GL_DEBUG_OUTPUT);
		glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
		glDebugMessageCallback(gl_debug_output, nullptr);
		glDebugMessageControl(GL_DONT_CARE, GL_DEBUG_TYPE_ERROR, GL_DONT_CARE, 0, nullptr, GL_TRUE);
	}

	// Set OpenGL global state
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);
	glFrontFace(GL_CCW);

	glClearColor(0.6, 0.48, 0.36, 1.f);

	// Setup ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
	//io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 450");

	// Initialize config
	InitializeSimulationConfig(gSimCfg, cuDeviceProp);
	logDebug("Simulation Config:\n");
	printf("    NumParticles : %zu\n", gSimCfg.NumParticles);
	printf("    GridDim      : (%u, %u, %u)\n", gSimCfg.GridDim.x, gSimCfg.GridDim.y, gSimCfg.GridDim.z);
	printf("    BlockDim     : (%u, %u, %u)\n", gSimCfg.BlockDim.x, gSimCfg.BlockDim.y, gSimCfg.BlockDim.z);
	printf("    Region       : (%f, %f, %f)\n", gSimCfg.Region.x, gSimCfg.Region.y, gSimCfg.Region.z);

	windowData.fixedDeltaTime = gSimCfg.PhysicsDeltaTime;

	// Initialize scene
	gScene.cameraPosition = glm::vec3{ 0, 0, gSimCfg.RegionHalf.x * (1 + sqrtf(2)) + gSimCfg.RegionHalf.z + 1.f };
	gScene.particleSize   = gSimCfg.ParticleRadius;
	gScene.displayMode    = DM_Particles;
	gScene.region         = *(glm::vec3*)&gSimCfg.Region;

	// Initialize OpenGL Renderer
	Renderer renderer {};
	if (FLSIM_SUCCESS != renderer.init()) {
		logError("Could not initialize renderer!");
		glfwDestroyWindow(window);
		glfwTerminate();
		return -1;
	}

	// Initialize Simulation state
	Simulation simulation {};
	if (FLSIM_SUCCESS != simulation.init(renderer)) {
		logError("Could not initialize simulation!");
		renderer.destroy();
		glfwDestroyWindow(window);
		glfwTerminate();
		return -1;
	}

	// Control variables
	SimulationMode simulationMode = SM_SingleStep;
	bool shouldSimulateNextStep = false;
	size_t stepCount = 0;

	float deltaTimeAccumulator = 0.f;

	// Main loop
	while (!glfwWindowShouldClose(window)) {
		// Check for events
		glfwPollEvents();

		// Start ImGui frame
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		// Update globals
		windowData.currentFrameTime = glfwGetTime();
		windowData.deltaTime        = windowData.currentFrameTime - windowData.lastFrameTime;
		windowData.lastFrameTime    = windowData.currentFrameTime;
		deltaTimeAccumulator += windowData.deltaTime;
		float framePerSecond = 1.f / windowData.deltaTime;
		if (simulationMode == SM_Playthrough || shouldSimulateNextStep) {
			windowData.fixedTotalTime += windowData.fixedDeltaTime;
		}

		// Update inputs
		{
			auto cameraDirection = glm::normalize(-gScene.cameraPosition);
			gScene.cameraPosition += cameraDirection * (5.f * windowData.deltaTime * windowData.scrollDelta.y);
		}

		// Update simulation
		if (simulationMode == SM_Playthrough || shouldSimulateNextStep) {
			simulation.update(windowData.fixedDeltaTime, windowData.fixedTotalTime);
			++stepCount;
		}

		// Render
		// if (deltaTimeAccumulator > gSimCfg.RenderDeltaTime)
		{
			// Clear framebuffer
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			// Render scene
			renderer.draw(gScene);
			//deltaTimeAccumulator -= gSimCfg.RenderDeltaTime;
		}

		shouldSimulateNextStep = false;

		// ImGui elements
		//ImGui::ShowDemoWindow();
		if (ImGui::Begin("Settings")) {
			if (ImGui::Selectable("Play", simulationMode == SM_Playthrough, 0, {40, 20})) simulationMode = SM_Playthrough;
			if (ImGui::Selectable("Pause", simulationMode == SM_SingleStep, 0, {40, 20})) simulationMode = SM_SingleStep ;

			if (simulationMode != SM_SingleStep) {
				ImGui::BeginDisabled();
				ImGui::Selectable("Step", false, 0, {40, 20});
				ImGui::EndDisabled();
			}
			else {
				if (ImGui::Selectable("Step", false, 0, {40, 20})) {
					shouldSimulateNextStep = true;
				}
			}
			ImGui::Combo("Display Mode", (int*)&gScene.displayMode, DisplayModeStr, DM_COUNT);
			ImGui::Separator();
			ImGui::InputScalar("Frames", ImGuiDataType_U64, &stepCount, 0, 0, 0, ImGuiInputTextFlags_ReadOnly);
			ImGui::InputScalar("Frame Time", ImGuiDataType_Float, &windowData.deltaTime, 0, 0, "%0.3f", ImGuiInputTextFlags_ReadOnly);
			ImGui::InputScalar("FPS", ImGuiDataType_Float, &framePerSecond, 0, 0, "%0.3f", ImGuiInputTextFlags_ReadOnly);
			ImGui::Separator();
			if (ImGui::Button("Dump data")) {
				simulation.dumpData();
			}

		}
		ImGui::End();

		// Render ImGui
		ImGui::Render();
		if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
			ImGui::UpdatePlatformWindows();
			ImGui::RenderPlatformWindowsDefault();
		}
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		// Display new frame
		glfwSwapBuffers(window);

		// Reset window data values until new events
		windowData.scrollDelta = { 0.f, 0.f };
	}

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

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
	if (!ImGui::GetIO().WantCaptureMouse) {
		if (button == GLFW_MOUSE_BUTTON_LEFT) {
			windowData.mouseDragging = (action == GLFW_PRESS);
		}
	}
}

void glfw_scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	if (!ImGui::GetIO().WantCaptureMouse) {
		windowData.scrollDelta.x = xoffset;
		windowData.scrollDelta.y = yoffset;
	}
}

void glfw_cursor_pos_callback(GLFWwindow *window, double xpos, double ypos)
{
	if (!ImGui::GetIO().WantCaptureMouse) {
		glm::vec2 currMousePosition{ xpos, ypos };
		windowData.mouseMovement = currMousePosition - windowData.prevMousePosition;
		windowData.prevMousePosition = currMousePosition;

		if (windowData.mouseDragging) {
			auto angles = windowData.mouseMovement * 5.f * windowData.fixedDeltaTime * -1.f;
			auto transform = glm::rotate(glm::mat4(1.f), angles.y, glm::vec3{ 1, 0, 0 });
			transform *= glm::rotate(glm::mat4(1.f), angles.x, glm::vec3{ 0, 1, 0 });
			gScene.cameraPosition = transform * glm::vec4{ gScene.cameraPosition, 1 };
		}
	}
}

void gl_debug_output(GLenum source, GLenum type, unsigned id, GLenum severity, GLsizei length, const char* message, const void* userParam)
{
	// ignore non-significant error/warning codes
    if(id == 131169 || id == 131185 || id == 131218 || id == 131204) return; 

    printf("\n---------------\n");
    printf("Debug message (%d): %s\n", id, message);

    switch (source)
    {
        case GL_DEBUG_SOURCE_API:             printf("Source: API\n"); break;
        case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   printf("Source: Window System\n"); break;
        case GL_DEBUG_SOURCE_SHADER_COMPILER: printf("Source: Shader Compiler\n"); break;
        case GL_DEBUG_SOURCE_THIRD_PARTY:     printf("Source: Third Party\n"); break;
        case GL_DEBUG_SOURCE_APPLICATION:     printf("Source: Application\n"); break;
        case GL_DEBUG_SOURCE_OTHER:           printf("Source: Other\n"); break;
    }

    switch (type)
    {
        case GL_DEBUG_TYPE_ERROR:               printf("Type: Error\n"); break;
        case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: printf("Type: Deprecated Behaviour\n"); break;
        case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  printf("Type: Undefined Behaviour\n"); break; 
        case GL_DEBUG_TYPE_PORTABILITY:         printf("Type: Portability\n"); break;
        case GL_DEBUG_TYPE_PERFORMANCE:         printf("Type: Performance\n"); break;
        case GL_DEBUG_TYPE_MARKER:              printf("Type: Marker\n"); break;
        case GL_DEBUG_TYPE_PUSH_GROUP:          printf("Type: Push Group\n"); break;
        case GL_DEBUG_TYPE_POP_GROUP:           printf("Type: Pop Group\n"); break;
        case GL_DEBUG_TYPE_OTHER:               printf("Type: Other\n"); break;
    }
    
    switch (severity)
    {
        case GL_DEBUG_SEVERITY_HIGH:         printf("Severity: high\n"); break;
        case GL_DEBUG_SEVERITY_MEDIUM:       printf("Severity: medium\n"); break;
        case GL_DEBUG_SEVERITY_LOW:          printf("Severity: low\n"); break;
        case GL_DEBUG_SEVERITY_NOTIFICATION: printf("Severity: notification\n"); break;
    }
}
