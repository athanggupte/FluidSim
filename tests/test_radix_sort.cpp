#include <cstdio>
#include <cuda_runtime.h>

#include "../src/config.h"
#include "../src/radix_sort.h"

int main()
{
	SimulationConfig config;
	config.NumParticles = 10000;
	config.RadixSortBlockSize = 128;

	uint32_t * h_data = new uint32_t[config.NumParticles];
	uint32_t * d_data;
	uint32_t * d_result;

	cudaMalloc(&d_data, config.NumParticles * sizeof(uint32_t));
	cudaMalloc(&d_result, config.NumParticles * sizeof(uint32_t));

	for (size_t i = 0; i < config.NumParticles; i++)
	{
		h_data[i] = rand() % 1000;
		printf("%d ", h_data[i]);
		if (i % config.RadixSortBlockSize == config.RadixSortBlockSize - 1)
			printf("\n");
	}

	cudaMemcpy(d_data, h_data, config.NumParticles * sizeof(uint32_t), cudaMemcpyHostToDevice);

	initializeRadixSort(config);
	radixSort(d_result, d_data, config.NumParticles);

	printf("---------------- sorted ----------------\n");

	cudaMemcpy(h_data, d_result, config.NumParticles * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	for (size_t i = 0; i < config.NumParticles; i++)
	{
		printf("%d ", h_data[i]);
		if (i % config.RadixSortBlockSize == config.RadixSortBlockSize - 1)
			printf("\n");
	}

	return 0;
}