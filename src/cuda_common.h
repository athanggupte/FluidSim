#pragma once

#include <curand_kernel.h>

#define cudaCall(fn, ...) \
	cuError = fn##(##__VA_ARGS__); \
	if (cudaSuccess != cuError) { \
		logError("[" STR(fn) "] Failed! error : [%d] %s :: %s", cuError, cudaGetErrorName(cuError), cudaGetErrorString(cuError)); \
		return FLSIM_ERROR; \
	}

#define cudaKernelCall(kernel, gridDim, blockDim, ...) \
	kernel <<< gridDim, blockDim >>> (##__VA_ARGS__); \
	cuError = cudaGetLastError(); \
	if (cudaSuccess != cuError) { \
		logError("Kernel call [" STR(kernel) "] Failed! error : [%d] %s :: %s", cuError, cudaGetErrorName(cuError), cudaGetErrorString(cuError)); \
		return FLSIM_ERROR; \
	}

#define cudaKernelCallShared(kernel, gridDim, blockDim, sharedMemSize, ...) \
	kernel <<< gridDim, blockDim, sharedMemSize >>> (##__VA_ARGS__); \
	cuError = cudaGetLastError(); \
	if (cudaSuccess != cuError) { \
		logError("Kernel call [" STR(kernel) "] Failed! error : [%d] %s :: %s", cuError, cudaGetErrorName(cuError), cudaGetErrorString(cuError)); \
		return FLSIM_ERROR; \
	}