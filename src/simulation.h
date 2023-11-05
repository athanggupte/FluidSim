#pragma once
#include <cuda_runtime.h>

// #include <glm/glm.hpp>

// #define NMAX 1000  /* Max number of particles */
// #define RCUT 2.5   /* Potential cut-off length */

// struct SimulationDesc
// {
// 	glm::ivec3 NUnitCell;
// 	float      Density;
// 	float      InitTemp;
// 	float      DeltaTime;
// 	int        StepLimit;
// 	int        StepAvg;
// };

// struct SimulationState
// {
// 	// Constants
// 	glm::vec3   Region;
// 	glm::vec3   RegionH;
// 	float       DensityH;

// 	// Variables
// 	int         nParticles;
// 	glm::vec3   p[NMAX];
// 	glm::vec3   v[NMAX];
// 	float       kinEnergy;
// 	float       potEnergy;
// 	float       totEnergy;
// 	int         stepCount;


// };

#define NMAX 1000
extern float positions[NMAX][3];

__global__ void cuSimulateParticles(float *positions, int count, float deltaTime, float totalTime);

void simulateParticles(float *positions, int count, float deltaTime, float totalTime);
