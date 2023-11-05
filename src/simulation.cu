#include <cuda_runtime.h>
#include <cmath>

__global__ void cuSimulateParticles(float * positions, int count, float deltaTime, float totalTime)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (tx < count) {
		float frequency = 2.0f;
		float amplitude = 0.4f;
		positions[3*tx] += amplitude * std::cos(frequency * totalTime) * deltaTime;
	}
}

void simulateParticles(float *positions, int count, float deltaTime, float totalTime)
{
	int numBlocks = 10;
	int numThreadsPerBlock = count / numBlocks;
	cuSimulateParticles <<<numBlocks, numThreadsPerBlock>>> (positions, count, deltaTime, totalTime);
}
