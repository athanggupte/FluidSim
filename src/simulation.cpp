#include "simulation.h"

#include <glad/glad.h>
#include <cuda_gl_interop.h>

#include "common.h"

thread_local static cudaError_t cuError;

Result Simulation::init(Renderer& renderer)
{
	size_t size;

	cudaCall(cudaGraphicsGLRegisterBuffer, &particlesGLCudaResource, renderer.instanceBuffer, cudaGraphicsRegisterFlagsNone);
	cudaCall(cudaGraphicsMapResources, 1, &particlesGLCudaResource);
	cudaCall(cudaGraphicsResourceGetMappedPointer, (void**)&positions, &size, particlesGLCudaResource);
	if (FLSIM_SUCCESS != __initializeParticles()) {
		return FLSIM_ERROR;
	}
	cudaCall(cudaGraphicsUnmapResources, 1, &particlesGLCudaResource);


	return FLSIM_SUCCESS;
}

Result Simulation::update(float deltaTime, float totalTime)
{
	size_t size;

	cudaCall(cudaGraphicsMapResources, 1, &particlesGLCudaResource);
	cudaCall(cudaGraphicsResourceGetMappedPointer, (void**)&positions, &size, particlesGLCudaResource);
	
	__simulateParticles(deltaTime, totalTime);

	cudaCall(cudaGraphicsUnmapResources, 1, &particlesGLCudaResource);

	return FLSIM_SUCCESS;
}

void writeArrayToFileFloat3(FILE * fp, float3 * arr, size_t count)
{
	for (size_t i = 0; i < count; i++) {
		fprintf(fp, "%e,%e,%e\n", arr[i].x, arr[i].y, arr[i].z);
	}
}

void writeArrayToFileFloat(FILE * fp, float * arr, size_t count)
{
	for (size_t i = 0; i < count; i++) {
		fprintf(fp, "%e\n", arr[i]);
	}
}

Result Simulation::dumpData()
{
	logDebug("Dumping data ...");

	size_t size;

	cudaCall(cudaGraphicsMapResources, 1, &particlesGLCudaResource);
	cudaCall(cudaGraphicsResourceGetMappedPointer, (void**)&positions, &size, particlesGLCudaResource);

	FILE * fp;
	void *tmp; float3 *pos; float3 *vel; float *dens; float *pres; float3 *acc;

	size_t sizeInBytes = (3 * sizeof(float3) + 2 * sizeof(float)) * gSimCfg.NumParticles;
	tmp = malloc(sizeInBytes);
	cudaMemcpy(tmp, positions, sizeInBytes, cudaMemcpyDeviceToHost);

	pos  = (float3*)tmp;
	vel  = (float3*)&pos[gSimCfg.NumParticles];
	dens = (float*)&vel[gSimCfg.NumParticles];
	pres = (float*)&dens[gSimCfg.NumParticles];
	acc  = (float3*)&pres[gSimCfg.NumParticles];

	//// All data
	//fp = fopen("flsim_datadump.dat", "w");
	//writeArrayToFileFloat(fp, (float*)tmp, (3 + 3 + 1 + 1) * gSimCfg.NumParticles);
	//fclose(fp);

	// Positions
	fp = fopen("flsim_pos.dat", "w");
	writeArrayToFileFloat3(fp, pos, gSimCfg.NumParticles);
	fclose(fp);
	logDebug("Finished writing positions to \"flsim_pos.dat\"");

	// Velocities
	fp = fopen("flsim_vel.dat", "w");
	writeArrayToFileFloat3(fp, vel, gSimCfg.NumParticles);
	fclose(fp);
	logDebug("Finished writing velocities to \"flsim_vel.dat\"");

	// Densities
	fp = fopen("flsim_dens.dat", "w");
	writeArrayToFileFloat(fp, dens, gSimCfg.NumParticles);
	fclose(fp);
	logDebug("Finished writing densities to \"flsim_dens.dat\"");

	// Pressures
	fp = fopen("flsim_pres.dat", "w");
	writeArrayToFileFloat(fp, pres, gSimCfg.NumParticles);
	fclose(fp);
	logDebug("Finished writing pressures to \"flsim_pres.dat\"");

	// Accelerations
	fp = fopen("flsim_acc.dat", "w");
	writeArrayToFileFloat3(fp, acc, gSimCfg.NumParticles);
	fclose(fp);
	logDebug("Finished writing accelerations to \"flsim_acc.dat\"");

	free(tmp);

	cudaCall(cudaGraphicsUnmapResources, 1, &particlesGLCudaResource);

	logDebug("Data dump Successful!");

	return FLSIM_SUCCESS;
}

