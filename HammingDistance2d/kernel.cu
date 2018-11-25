#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#include <string>
#include <cstdio>
#include <vector>
#include <iterator>
#include <iostream>
#include <random>
#include <chrono>
#include <memory>
#include <functional>

#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<conio.h>


using namespace std;

#define N 10000 //rozmiar ci¹gu binarnergo
#define M 100 //iloœæ tablic binarnych
#define DIST 2 //szukany dystans
#define BLOCKSIZE_x 16
#define BLOCKSIZE_y 16

#define CUDA_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

int iDivUp(int hostPtr, int b) { return ((hostPtr % b) != 0) ? (hostPtr / b + 1) : (hostPtr / b); }

__global__ void cudaHammingDistance2d(bool *devPtr, size_t pitch)
{
	int tidx = blockIdx.x*blockDim.x + threadIdx.x;
	int tidy = blockIdx.y*blockDim.y + threadIdx.y;

	if ((tidx < N) && (tidy < M))
	{
		/*float *row_a = (float *)((char*)devPtr + tidy * pitch);
		row_a[tidx] = row_a[tidx] * tidx * tidy;*/
	}
}

static int m_w = 17358;
static int m_z = 341;

__host__ unsigned int simplerand(void) {
	m_z = 36969 * (m_z & 65535) + (m_z >> 16);
	m_w = 18000 * (m_w & 65535) + (m_w >> 16);
	return (m_z << 16) + m_w;
}

__host__ void initialize_arrays(bool** arrays, bool randomDistance = false)
{
	std::minstd_rand gen(std::random_device{}());
	std::uniform_real_distribution<double> dist(0, 1);
	if (randomDistance)
		for (long long i = 0; i < M; ++i)
			for (long long j = 0; j < N; ++j)
				arrays[i][j] = (((double)simplerand() + UINT_MAX) / UINT_MAX / 2) > 0.5;
	else
	{
		int* indexes = (int*)calloc(M, sizeof(int));
		int ind = 1;
		while (ind < M)
		{
			bool flag = 0;
			int rand = (int)((((double)simplerand() + UINT_MAX) / UINT_MAX / 2) * N);
			for (int i = 0; i < M; i++)
				if (indexes[i] == rand)
				{
					flag = 1;
					break;
				}
			if (flag)
				continue;
			indexes[ind++] = rand;
		}
		for (int i = 0; i < M; i++)
			arrays[i][indexes[i]] = true;
		free(indexes);
	}
}

__host__ bool CpuHammingDistance2d(bool **bitArrays)
{
	bool returnFlag = true;
	auto start = chrono::high_resolution_clock::now();
	int** distances = (int**)calloc(M, sizeof(int*));
	for (int i = 0; i < M; i++)
		distances[i] = (int*)calloc(M, sizeof(int));
	for (int i = 0; i < M; i++)
		for (int j = 0; j < M; j++)
		{
			if (i == j)
				break;
			for (int p = 0; p < N; p++)
				if (bitArrays[i][p] != bitArrays[j][p])
					distances[i][j]++;
		}
	for (int i = 0; i < M; i++)
		for (int j = 0; j < M; j++)
		{
			if (i == j)
				break;
			if (distances[i][j] != DIST)
				returnFlag = false;
		}
	for (int i = 0; i < M; i++)
		free(distances[i]);
	free(distances);
	auto finish = chrono::high_resolution_clock::now();
	auto miliseconds = chrono::duration_cast<chrono::milliseconds>(finish - start);
	cout << "CpuHammingDistance2d time: " << miliseconds.count() << " milliseconds\n";
	cout << "Result: " << returnFlag << "\n";
	return returnFlag;
}

__host__ bool GpuHammingDistance2d(bool **sss)
{
	bool bitArrays[M][N];
	bool *d_bitArrays;
	size_t pitch;

	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++) {
			bitArrays[i][j] = false;
			//printf("row %i column %i value %f \n", i, j, hostPtr[i][j]);
		}

	CUDA_CALL(cudaSetDevice(0));
	CUDA_CALL(cudaMallocPitch(&d_bitArrays, &pitch, N * sizeof(bool), M));
	CUDA_CALL(cudaMemcpy2D(d_bitArrays, pitch, bitArrays, N * sizeof(bool), N * sizeof(bool), M, cudaMemcpyHostToDevice));
	dim3 gridSize(iDivUp(N, BLOCKSIZE_x), iDivUp(M, BLOCKSIZE_y));
	dim3 blockSize(BLOCKSIZE_y, BLOCKSIZE_x);
	////size_t free = 0, total = 0;
	////cudaMemGetInfo(&free, &total);
	auto start = chrono::high_resolution_clock::now();
	cudaHammingDistance2d << <gridSize, blockSize >> > (d_bitArrays, pitch);

	// Check for any errors launching the kernel
	CUDA_CALL(cudaPeekAtLastError());

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	CUDA_CALL(cudaDeviceSynchronize());


	auto finish = chrono::high_resolution_clock::now();
	auto milliseconds = chrono::duration_cast<chrono::milliseconds>(finish - start);
	cout << "GPU time: " << milliseconds.count() << " milliseconds\n";
	return true;
}

__host__ int main()
{
	auto start = chrono::high_resolution_clock::now();
	bool** bitArrays = (bool**)malloc(M * sizeof(bool*));
	for (int i = 0; i < M; i++)
		bitArrays[i] = (bool*)calloc(N, sizeof(bool));
	auto finish = chrono::high_resolution_clock::now();
	auto miliseconds = chrono::duration_cast<chrono::milliseconds>(finish - start);
	cout << "Malloc time: " << miliseconds.count() << " milliseconds\n";

	start = chrono::high_resolution_clock::now();
	initialize_arrays(bitArrays, false);
	finish = chrono::high_resolution_clock::now();
	miliseconds = chrono::duration_cast<chrono::milliseconds>(finish - start);
	cout << "initialize_arrays time: " << miliseconds.count() << " milliseconds\n";
	cout << "Arrays length: " << N << " number of arrays: " << M << "\n";

	//CpuHammingDistance2d(bitArrays);
	GpuHammingDistance2d(bitArrays);

	for (int i = 0; i < M; i++)
		free(bitArrays[i]);
	free(bitArrays);
	return 0;
}
