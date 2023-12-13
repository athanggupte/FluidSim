#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "config.h"
#include "radix_sort.h"

__global__ void cuApplyOrder(float* d_out_values, uint32_t* d_indexes, size_t count);

int main()
{
	SimulationConfig config;
	config.NumParticles = 10;
	config.RadixSortBlockSize = 4;

	uint32_t *h_keys = new uint32_t[config.NumParticles];
	float3 *h_values = new float3[config.NumParticles];
	uint32_t *d_keys;
	float3 *d_values;
	uint32_t *d_indexes;

	cudaMalloc(&d_keys, config.NumParticles * sizeof(uint32_t));
	cudaMalloc(&d_values, config.NumParticles * sizeof(float3));

	for (size_t i = 0; i < config.NumParticles; i++)
	{
		h_keys[i] = rand() % 1000;
		h_values[i] = float3(h_keys[i] * 0.1f, 0, 0);
		printf("%d (%f, %f, %f) ", h_keys[i], h_values[i].x, h_values[i].y, h_values[i].z);
		//if (i % config.RadixSortBlockSize == config.RadixSortBlockSize - 1)
		printf("\n");
	}

	cudaMemcpy(d_keys, h_keys, config.NumParticles * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_values, h_values, config.NumParticles * sizeof(float3), cudaMemcpyHostToDevice);

	initializeRadixSort(config);
	radixSort(d_keys, &d_indexes, config.NumParticles);

	__applyOrder(d_values, d_indexes, config.NumParticles);

	printf("---------------- sorted ----------------\n");

	cudaMemcpy(h_keys, d_keys, config.NumParticles * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_values, d_values, config.NumParticles * sizeof(float3), cudaMemcpyDeviceToHost);
	for (size_t i = 0; i < config.NumParticles; i++)
	{
		printf("%d (%f, %f, %f) ", h_keys[i], h_values[i].x, h_values[i].y, h_values[i].z);
		//if (i % config.RadixSortBlockSize == config.RadixSortBlockSize - 1)
			printf("\n");
	}

	//printf("---------------- indexes :\n");
	//cudaMemcpy(h_data, d_indexes, config.NumParticles * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	//for (size_t i = 0; i < config.NumParticles; i++)
	//{
	//	printf("%d ", h_data[i]);
	//}

	return 0;
}
