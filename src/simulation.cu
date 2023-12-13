#include "simulation.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "config.h"

#ifdef __INTELLISENSE__
void __syncthreads();
#endif

#include <cmath>
#include <glm/geometric.hpp>
#include <glm/gtc/constants.hpp>

#include "cuda_common.h"
#include "radix_sort.h"

struct SimulationData
{
	float3 * positions;
	float3 * velocities;
	float * densities;
	float * pressures;
	float3 * accelerations;
};

// CUDA Constants
__constant__ __device__ struct SimulationConfig dSimCfg;
curandState_t * gRngStates;

// CUDA Kernel Function Forward Declarations
__global__ void cuInitializeParticles(SimulationData sim, curandState_t * rngStates, size_t count);
__global__ void cuSimulateParticles(SimulationData sim, curandState_t * rngStates, size_t count, float deltaTime, float totalTime);

thread_local static cudaError_t cuError;

// Host calling functions
Result Simulation::__initializeParticles()
{
	cudaCall(cudaMemcpyToSymbol, dSimCfg, &gSimCfg, sizeof(SimulationConfig));

	cudaCall(cudaMalloc, &gRngStates, sizeof(curandState_t) * gSimCfg.NumParticles);

	cudaCall(cudaMalloc, &cellStarts, sizeof(uint32_t) * gSimCfg.NumCells);
	cudaCall(cudaMalloc, &particleCellIdx, sizeof(uint32_t) * gSimCfg.NumParticles);

#ifdef FLSIM_CHECK_DEV_SIMCFG
	SimulationConfig tmpCfg;
	cudaMemcpyFromSymbol(&tmpCfg, dSimCfg, sizeof(SimulationConfig));
	printf("NumParticles: %zu\n", tmpCfg.NumParticles);
	printf("ThreadGridDim: (%u, %u, %u)\n", tmpCfg.ThreadGridDim.x, tmpCfg.ThreadGridDim.y, tmpCfg.ThreadGridDim.z);
	printf("ThreadBlockDim: (%u, %u, %u)\n", tmpCfg.ThreadBlockDim.x, tmpCfg.ThreadBlockDim.y, tmpCfg.ThreadBlockDim.z);
	printf("Region: (%f, %f, %f)\n", tmpCfg.Region.x, tmpCfg.Region.y, tmpCfg.Region.z);
#endif

	velocities    = (float3*)&positions[gSimCfg.NumParticles];
	densities     = (float*)&velocities[gSimCfg.NumParticles];
	pressures     = (float*)&densities[gSimCfg.NumParticles];
	accelerations = (float3*)&pressures[gSimCfg.NumParticles];

	SimulationData inputs { positions, velocities, densities, pressures, accelerations };
	cudaKernelCall(cuInitializeParticles, gSimCfg.ThreadGridDim, gSimCfg.ThreadBlockDim, inputs, gRngStates, gSimCfg.NumParticles);

	return FLSIM_SUCCESS;
}

Result Simulation::__initializeCells()
{

	return FLSIM_SUCCESS;
}

Result Simulation::__simulateParticles(float deltaTime, float totalTime)
{
	SimulationData inputs { positions, velocities, densities, pressures, accelerations };
	cudaKernelCall(cuSimulateParticles, gSimCfg.ThreadGridDim, gSimCfg.ThreadBlockDim, inputs, gRngStates, gSimCfg.NumParticles, deltaTime, totalTime);

	return FLSIM_SUCCESS;
}

Result Simulation::__updateCells()
{
	return FLSIM_SUCCESS;
}


/***************************************************************
 * CUDA Kernel Function Definitions
 **************************************************************/

__device__ uint32_t cuGlobalIndex()
{
	uint32_t threadNumInBlock = threadIdx.x * (blockDim.y * blockDim.z) + threadIdx.y * (blockDim.z) + threadIdx.z;
    uint32_t blockNumInGrid = blockIdx.x * (gridDim.y * gridDim.z) + blockIdx.y * (gridDim.z) + blockIdx.z;
    uint32_t threadsPerBlock = blockDim.x * blockDim.y * blockDim.z;

	return threadNumInBlock + blockNumInGrid * threadsPerBlock;
}

__device__ dim3 cuVectorIndex(int gid)
{
	dim3 NThreads = dim3(gridDim.x * blockDim.x, gridDim.y * blockDim.y, gridDim.z * blockDim.z);

	uint32_t xidx = gid / (NThreads.y * NThreads.z);
	uint32_t yidx = (gid / NThreads.z) % NThreads.y;
	uint32_t zidx = gid % NThreads.z;

	/*uint32_t xidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t yidx = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t zidx = threadIdx.z + blockIdx.z * blockDim.z;*/
    return { xidx, yidx, zidx };
}

__device__ dim3 cuCellVectorIndex(float3 position)
{
	float3 cellSize = dSimCfg.CellSize;
	float3 regionHalf = dSimCfg.RegionHalf;

	uint32_t xidx = (position.x + regionHalf.x) / cellSize.x;
	uint32_t yidx = (position.y + regionHalf.y) / cellSize.y;
	uint32_t zidx = (position.z + regionHalf.z) / cellSize.z;

	return { xidx, yidx, zidx };
}

__device__ uint32_t cuCellIndex(dim3 vid)
{
	return vid.x * dSimCfg.CellGridDim.y * dSimCfg.CellGridDim.z + vid.y * dSimCfg.CellGridDim.z + vid.z;
}

__device__ void cuOscillate(float& val, float deltaTime, float totalTime)
{
	float frequency = 2.0f;
	float amplitude = 2.4f;
	val += amplitude * std::cos(frequency * totalTime) * deltaTime;
}

__device__ glm::vec3 cuRandomDirection(curandState_t * rngState)
{
	float theta = curand_uniform(rngState) * glm::two_pi<float>();
	float phi   = curand_uniform(rngState) * glm::pi<float>();

	glm::vec3 randVec;
	randVec.x = sin(phi) * cos(theta);
	randVec.y = sin(phi) * sin(theta);
	randVec.z = cos(phi);

	return randVec;
}

__device__ float cuWendlandC2(float h, float r)
{
	float alpha = 21.f / (16.f * glm::pi<float>() * h * h * h);
	float WC2 = 0.f;
	float q = r / h;

	if (q < 2.f) {
		float term1 = 1 - q / 2.f;
		term1 = term1 * term1;
		term1 = term1 * term1;

		float term2 = 2 * q + 1;

		WC2 = alpha * term1 * term2;
	}

	return WC2;
}

__device__ float cuWendlandC2Derivative(float h, float r)
{
	float alpha = -105.f / (16.f * glm::pi<float>() * h * h * h);
	float WC2 = 0.f;
	float q = r / h;

	if (q < 2.f) {
		if (r > 1e-8f) {
			float term1 = 1 - q / 2.f;
			term1 = term1 * term1 * term1;

			WC2 = alpha * term1 * q;
		}
	}

	return WC2;
}

__device__ void cuResolveCollisions(glm::vec3& position, glm::vec3& velocity)
{
	if (fabsf(position.x) > dSimCfg.RegionHalf.x) {
		position.x = copysignf(dSimCfg.RegionHalf.x, position.x);
		velocity.x *= -1 * dSimCfg.CollisionDamping;
	}
	if (fabsf(position.y) > dSimCfg.RegionHalf.y) {
		position.y = copysignf(dSimCfg.RegionHalf.y, position.y);
		velocity.y *= -1 * dSimCfg.CollisionDamping;
	}
	if (fabsf(position.z) > dSimCfg.RegionHalf.z) {
		position.z = copysignf(dSimCfg.RegionHalf.z, position.z);
		velocity.z *= -1 * dSimCfg.CollisionDamping;
	}
}

__device__ float cuCalculateDensity(float3 * positions, size_t count, uint32_t gid)
{
	float density = 0.f;

	glm::vec3 position = *(glm::vec3*)&positions[gid];

	for (uint32_t i = 0; i < count; i++) {
		float dist = glm::distance(*(glm::vec3*)&positions[i], position);

		if (dist < 2 * dSimCfg.SmoothingRadius) {
			float W = cuWendlandC2(dSimCfg.SmoothingRadius, dist);
			density += dSimCfg.ParticleMass * W;
		}
	}

	return density;
}

__device__ float cuCalculatePressure(float density)
{
	float pressure = dSimCfg.ReferencePressure * (powf(density / dSimCfg.ReferenceDensity, dSimCfg.TaitExponent) - 1);
	//return __max(pressure, 0.00001f);
	return pressure;
}

__device__ glm::vec3 cuCalculatePressureAcceleration(SimulationData sim, curandState_t * rngStates, size_t count, uint32_t gid)
{
	glm::vec3 pressureAcc { 0.f };

	glm::vec3 position = *(glm::vec3*)&sim.positions[gid];
	glm::vec3 velocity = *(glm::vec3*)&sim.velocities[gid];
	float     density  = sim.densities[gid];
	float     pressure = sim.pressures[gid];
	float     density2 = density * density;

	density += glm::epsilon<float>();
	density2 += glm::epsilon<float>();

	for (uint32_t i = 0; i < count; i++) {
		if (i == gid) continue;
		glm::vec3 deltaPos = *(glm::vec3*)&sim.positions[i] - position;
		float dist = glm::length(deltaPos);

		if (dist < 2 * dSimCfg.SmoothingRadius) {
			float dW = cuWendlandC2Derivative(dSimCfg.SmoothingRadius, dist);

			glm::vec3 dir = dist < 1e-8f ? cuRandomDirection(&rngStates[gid]) : deltaPos / dist;
			glm::vec3 delW = dir * dW;

			pressureAcc += density * dSimCfg.ParticleMass * ((pressure / density2) + (sim.pressures[i] / (sim.densities[i] * sim.densities[i]))) * delW;
			// pressureAcc += - dSimCfg.ParticleMass * sim.pressures[i] / sim.densities[i] * delW;
			// pressureAcc += dSimCfg.ParticleMass / density * (pressure + sim.pressures[i]) * delW;
		}
	}
	pressureAcc *= 1 / density;

	return pressureAcc;
}

__device__ glm::vec3 cuCalculateViscosityAcceleration(SimulationData sim, curandState_t * rngStates, size_t count, uint32_t gid)
{
	glm::vec3 viscosityAcc { 0.f };

	glm::vec3 position = *(glm::vec3*)&sim.positions[gid];
	glm::vec3 velocity = *(glm::vec3*)&sim.velocities[gid];
	float     density  = sim.densities[gid];
	float     pressure = sim.pressures[gid];
	float     density2 = density * density;

	density += glm::epsilon<float>();
	density2 += glm::epsilon<float>();

	for (uint32_t i = 0; i < count; i++) {
		if (i == gid) continue;
		glm::vec3 deltaPos = *(glm::vec3*)&sim.positions[i] - position;
		glm::vec3 deltaVel = *(glm::vec3*)&sim.velocities[i] - velocity;
		float dist = glm::length(deltaPos);
		dist += glm::epsilon<float>();

		if (dist < 2 * dSimCfg.SmoothingRadius) {
			float dW = cuWendlandC2Derivative(dSimCfg.SmoothingRadius, dist);

			glm::vec3 dir = dist < 1e-8f ? cuRandomDirection(&rngStates[gid]) : deltaPos / dist;
			glm::vec3 delW = dir * dW;
			float normDelW = glm::length(delW);

			viscosityAcc += dSimCfg.ParticleMass / density * dSimCfg.Viscosity * 2.f * deltaVel * normDelW / dist;
		}
	}

	return viscosityAcc;
}

__global__ void cuInitializeParticles(SimulationData sim, curandState_t * rngStates, size_t count)
{
	dim3 globalDim = dim3(gridDim.x * blockDim.x, gridDim.y * blockDim.y, gridDim.z * blockDim.z);
	size_t nThreads = globalDim.x * globalDim.y * globalDim.z;
	uint32_t gid = cuGlobalIndex();
	dim3 vid = cuVectorIndex(gid);

	curand_init(53135, gid, 0, &rngStates[gid]);

	float3 noise      = { (curand_uniform(&rngStates[gid]) - 0.5f) / nThreads, (curand_uniform(&rngStates[gid]) - 0.5f) / nThreads, (curand_uniform(&rngStates[gid]) - 0.5f) / nThreads };
	float3 fraction   = { (float)vid.x / (float)globalDim.x, (float)vid.y / (float)globalDim.y, (float)vid.z / (float)globalDim.z };
	float3 distance   = { noise.x + fraction.x * dSimCfg.InitRegion.x, noise.y + fraction.y * dSimCfg.InitRegion.y, noise.z + fraction.z * dSimCfg.InitRegion.z };
	float3 position   = { distance.x + dSimCfg.InitPosition.x, distance.y + dSimCfg.InitPosition.y, distance.z + dSimCfg.InitPosition.z };

	if (gid < count) {
		sim.positions[gid] = { position.x, position.y, position.z };
		sim.velocities[gid] = { 0.f, 0.f, 0.f };

		__syncthreads();

		float density = cuCalculateDensity(sim.positions, count, gid);
		sim.densities[gid] = density;
		__syncthreads();

		sim.pressures[gid] = cuCalculatePressure(sim.densities[gid]);
		__syncthreads();

		glm::vec3 pressureAcc = cuCalculatePressureAcceleration(sim, rngStates, count, gid);
		glm::vec3 viscosityAcc = cuCalculateViscosityAcceleration(sim, rngStates, count, gid);

		glm::vec3 acceleration = pressureAcc + viscosityAcc + (*(glm::vec3*)&dSimCfg.Gravity);

		sim.accelerations[gid] = *(float3*)&acceleration;
	}
}

__global__ void cuSimulateParticles(SimulationData sim, curandState_t * rngStates, size_t count, float deltaTime, float totalTime)
{
	uint32_t gid = cuGlobalIndex();

	if (gid < count) {
		glm::vec3 prev_position     = *(glm::vec3*)&sim.positions[gid];
		glm::vec3 prev_velocity     = *(glm::vec3*)&sim.velocities[gid];

		glm::vec3 position     = prev_position;
		glm::vec3 velocity     = prev_velocity;
		float     density;
		float     pressure;

		glm::vec3 pressureAcc = cuCalculatePressureAcceleration(sim, rngStates, count, gid);
		glm::vec3 viscosityAcc = cuCalculateViscosityAcceleration(sim, rngStates, count, gid);
		glm::vec3 acceleration = pressureAcc + viscosityAcc + (*(glm::vec3*)&dSimCfg.Gravity);

		// half tap
		velocity += 0.5f * deltaTime * acceleration;
		position += 0.5f * deltaTime * velocity;
		cuResolveCollisions(position, velocity);
		density  = cuCalculateDensity(sim.positions, count, gid);
		pressure = cuCalculatePressure(density);

		sim.velocities[gid] = *(float3*)&velocity;
		sim.positions[gid]  = *(float3*)&position;
		sim.densities[gid]  = density;
		sim.pressures[gid]  = pressure;
		__syncthreads();

		pressureAcc = cuCalculatePressureAcceleration(sim, rngStates, count, gid);
		viscosityAcc = cuCalculateViscosityAcceleration(sim, rngStates, count, gid);
		acceleration = pressureAcc + viscosityAcc + (*(glm::vec3*)&dSimCfg.Gravity);

		// full tap
		velocity = prev_velocity + deltaTime * acceleration;
		position = prev_position + deltaTime * velocity;
		cuResolveCollisions(position, velocity);
		density  = cuCalculateDensity(sim.positions, count, gid);
		pressure = cuCalculatePressure(density);


		sim.velocities[gid]    = *(float3*)&velocity;
		sim.positions[gid]     = *(float3*)&position;
		sim.densities[gid]     = density;
		sim.pressures[gid]     = pressure;
		sim.accelerations[gid] = *(float3*)&acceleration; // for plotting only
	}
}




//for (uint32_t i = 0; i < count; i++) {
		//	if (i == gid) continue;
		//	glm::vec3 deltaPos = *(glm::vec3*)&positions[i] - position;
		//	float dist = glm::length(deltaPos);

		//	if (dist < 2 * dSimCfg.SmoothingRadius) {
		//		glm::vec3 deltaVel = *(glm::vec3*)&velocities[i] - velocity;

		//		float W = cuWendlandC2(dSimCfg.SmoothingRadius, dist);
		//		glm::vec3 delWx { cuWendlandC2(dSimCfg.SmoothingRadius, deltaPos.x), cuWendlandC2(dSimCfg.SmoothingRadius, deltaPos.y), cuWendlandC2(dSimCfg.SmoothingRadius, deltaPos.z) };

		//		newDensity += dSimCfg.ParticleMass * W;
		//		divVel  += dSimCfg.ParticleMass * glm::dot(deltaVel, delWx);

		//		pressureGradTerm += - dSimCfg.ParticleMass * ((pressure / density2) + (pressures[i] / (densities[i] * densities[i]))) * delWx;
		//		//viscosityTerm    += dSimCfg.Viscosity / pressure * dSimCfg.ParticleMass * 4 / ((pressure + pressures[i]) * glm::dot(deltaPos, deltaPos)) * (glm::dot(deltaPos, delWx)) * deltaVel;
		//	}
		//}