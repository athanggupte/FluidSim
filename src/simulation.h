#pragma once
#include <cuda_runtime.h>

#include "common.h"
#include "config.h"
#include "render.h"

enum SimulationMode
{
	SM_Playthrough,
	SM_SingleStep,

	SM_COUNT
};

constexpr const char* SimulationModeStr[SM_COUNT] = {
	"Playthrough",
	"Single Step",
};

enum SimulationAction
{
	SA_Play,
	SA_Pause,
	SA_Reset,
	SA_StepForward,
	SA_StepBackward,

	SA_COUNT
};

struct Simulation
{
	float3 * positions;
	float3 * velocities;
	float  * densities;
	float  * pressures;
	float3 * accelerations;
	cudaGraphicsResource_t particlesGLCudaResource;

	uint32_t * cellStarts;
	uint32_t * particleCellIdx;
	uint32_t * sortedIdxs;

	Result init(Renderer& renderer);
	Result update(float deltaTime, float totalTime);
	Result dumpData();

	Result __initializeParticles();
	Result __initializeCells();
	Result __simulateParticles(float deltaTime, float totalTime);
	Result __updateCells();
};

struct SimulationData
{
	float3 * positions;
	float3 * velocities;
	float  * densities;
	float  * pressures;
	float3 * accelerations;

	uint32_t * cellStarts;
	uint32_t * particleCellIdx;
};
