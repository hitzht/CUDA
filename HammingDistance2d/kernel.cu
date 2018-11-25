#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#include <string>
#include <stdio.h>
#include <vector>
#include <iterator>
#include <iostream>
#include <random>
#include <chrono>
#include <memory>
#include <functional>

using namespace std;

#define N 10000LL //rozmiar ci¹gu binarnergo
#define M 1000LL //iloœæ tablic binarnych

#define CUDA_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__host__ bool CpuHammingDistance2d(char **bitArrays, const long long arrayLength, const long long numberOfArray)
{
	return true;
}

__host__ bool GpuHammingDistance2d(char **bitArrays, const long long arrayLength, const long long numberOfArray)
{
	return true;
}

__host__ long main()
{
	long long arrayLength = N;
	long long numberOfArray = M;
	char** bitArrays;

	CpuHammingDistance2d(bitArrays, arrayLength, numberOfArray);
	GpuHammingDistance2d(bitArrays, arrayLength, numberOfArray);

	return 0;
}
