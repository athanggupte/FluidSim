#include "radix_sort.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "common.h"
#include "config.h"
#include "cuda_common.h"
#include "radix_sort/scan.h"
#include "radix_sort/sort.h"

extern __constant__ __device__ SimulationConfig const* dSimCfg;

thread_local static cudaError_t cuError;

uint32_t d_in_len;
uint32_t block_sz;
uint32_t max_elems_per_block;
uint32_t grid_sz;

uint32_t* d_prefix_sums;
uint32_t d_prefix_sums_len;

uint32_t* d_block_sums;
uint32_t d_block_sums_len;

uint32_t* d_scan_block_sums;

uint32_t* d_idxs;

uint32_t s_data_len;
uint32_t s_mask_out_len;
uint32_t s_merged_scan_mask_out_len;
uint32_t s_mask_out_sums_len;
uint32_t s_scan_mask_out_sums_len;
uint32_t shmem_sz;

Result initializeRadixSort(SimulationConfig const& config)
{
	d_in_len = config.NumParticles;
	block_sz = config.RadixSortBlockSize;
    max_elems_per_block = block_sz;
    grid_sz = d_in_len / max_elems_per_block;
    // Take advantage of the fact that integer division drops the decimals
    if (d_in_len % max_elems_per_block != 0)
        grid_sz += 1;

    d_prefix_sums_len = d_in_len;
    cudaCall(cudaMalloc, &d_prefix_sums, sizeof(uint32_t) * d_prefix_sums_len);
    //cudaCall(cudaMemset, d_prefix_sums, 0, sizeof(uint32_t) * d_prefix_sums_len);

    d_block_sums_len = 4 * grid_sz; // 4-way split
    cudaCall(cudaMalloc, &d_block_sums, sizeof(uint32_t) * d_block_sums_len);
    //cudaCall(cudaMemset, d_block_sums, 0, sizeof(uint32_t) * d_block_sums_len);

    cudaCall(cudaMalloc, &d_scan_block_sums, sizeof(uint32_t) * d_block_sums_len);
    //cudaCall(cudaMemset, d_scan_block_sums, 0, sizeof(uint32_t) * d_block_sums_len);

    cudaCall(cudaMalloc, &d_idxs, sizeof(uint32_t) * d_in_len);
    //cudaCall(cudaMemset, d_idxs, 0, sizeof(uint32_t) * d_in_len);

    // shared memory consists of 3 arrays the size of the block-wise input
    //  and 2 arrays the size of n in the current n-way split (4)
    s_data_len = max_elems_per_block;
    s_mask_out_len = max_elems_per_block + 1;
    s_merged_scan_mask_out_len = max_elems_per_block;
    s_mask_out_sums_len = 4; // 4-way split
    s_scan_mask_out_sums_len = 4;
    shmem_sz = (s_data_len + s_data_len
					+ s_mask_out_len
                    + s_merged_scan_mask_out_len
                    + s_mask_out_sums_len
                    + s_scan_mask_out_sums_len)
                    * sizeof(uint32_t);

	return FLSIM_SUCCESS;
}

__global__ void gpu_radix_sort_local(unsigned int* d_out_sorted, unsigned int* d_out_idxs,
    unsigned int* d_prefix_sums,
    unsigned int* d_block_sums,
    unsigned int input_shift_width,
    unsigned int* d_in,
    unsigned int d_in_len,
    unsigned int max_elems_per_block);

__global__ void gpu_glbl_shuffle(unsigned int* d_out, unsigned int* d_out_idxs,
    unsigned int* d_in,
    unsigned int* d_scan_block_sums,
    unsigned int* d_prefix_sums,
    unsigned int input_shift_width,
    unsigned int d_in_len,
    unsigned int max_elems_per_block);

__global__ void gpu_gen_indexes(uint32_t* d_idxs, uint32_t d_in_len)
{
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= d_in_len)
		return;

	d_idxs[idx] = idx;
}

Result radixSort(uint32_t* inputs, uint32_t** p_out_idxs, size_t n)
{
	uint32_t* d_in = inputs;
	uint32_t* d_out = inputs; // TODO: Check

    cudaKernelCall(gpu_gen_indexes, grid_sz, block_sz, d_idxs, d_in_len);

	// for every 2 bits from LSB to MSB:
    //  block-wise radix sort (write blocks back to global memory)
    for (unsigned int shift_width = 0; shift_width <= 30; shift_width += 2)
    {
        cudaKernelCallShared(gpu_radix_sort_local, grid_sz, block_sz, shmem_sz, d_out, d_idxs,
                                                                d_prefix_sums, 
                                                                d_block_sums, 
                                                                shift_width, 
                                                                d_in, 
                                                                d_in_len, 
                                                                max_elems_per_block);

        //unsigned int* h_test = new unsigned int[d_in_len];
        //checkCudaErrors(cudaMemcpy(h_test, d_in, sizeof(unsigned int) * d_in_len, cudaMemcpyDeviceToHost));
        //for (unsigned int i = 0; i < d_in_len; ++i)
        //    std::cout << h_test[i] << " ";
        //std::cout << std::endl;
        //delete[] h_test;

        // scan global block sum array
        sum_scan_blelloch(d_scan_block_sums, d_block_sums, d_block_sums_len);

        // scatter/shuffle block-wise sorted array to final positions
        cudaKernelCall(gpu_glbl_shuffle, grid_sz, block_sz, d_in, d_idxs,
                                                    d_out, 
                                                    d_scan_block_sums, 
                                                    d_prefix_sums, 
                                                    shift_width, 
                                                    d_in_len, 
                                                    max_elems_per_block);
    }
    //cudaCall(cudaMemcpy, d_out, d_in, sizeof(unsigned int) * d_in_len, cudaMemcpyDeviceToDevice);
    *p_out_idxs = d_idxs;

	return FLSIM_SUCCESS;
}

__global__ void cuApplyOrder(float3* d_out_values, uint32_t* d_indexes, size_t count)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= count)
		return;

	float3 tmp = d_out_values[d_indexes[idx]];
	__syncthreads();
	d_out_values[idx] = tmp;
}

Result __applyOrder(float3* d_out_values, uint32_t* d_indexes, size_t count)
{
	cudaKernelCall(cuApplyOrder, 100, 100, d_out_values, d_indexes, count);

    return FLSIM_SUCCESS;
}
