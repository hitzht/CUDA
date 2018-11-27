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
#include <Windows.h>

#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<conio.h>


using namespace std;

#define N 40 //rozmiar ci¹gu binarnergo
#define M 15000 //iloœæ tablic binarnyc h
#define SHOW_DIFFS true //czy pokazywaæ ró¿nicê miêdzy kolejnymi danymi ci¹gami bitów

#define CUDA_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__device__ unsigned long long counter = 0;

__global__ void cudaHammingDistance2dEquals1(bool *d_arrays, bool *d_pairs)
{
	const long numThreads = blockDim.x * gridDim.x;
	const long threadID = blockIdx.x * blockDim.x + threadIdx.x;
	int i, j;
	bool flag = false;
	for (int ind = threadID; ind < M * M; ind += numThreads)
	{
		i = ind % M;
		j = ind / M;
		if (i == j)
			continue;
		for (int p = 0; p < N; p += 1)
			if (d_arrays[i * N + p] != d_arrays[j * N + p])
				if (flag)
				{
					flag = false;
					break;
				}
				else
					flag = true;
		if (flag)
		{
			d_pairs[i * M + j] = true;
		}
	}
}

__global__ void cudaHammingDistance2d(bool *d_arrays, unsigned long long *d_distances)
{
	const long numThreads = blockDim.x * gridDim.x;
	const long threadID = blockIdx.x * blockDim.x + threadIdx.x;
	for (int ind = threadID; ind < M * M; ind += numThreads)
	{
		int i = ind % M;
		int j = ind / M;
		if (i == j)
			continue;
		for (int p = 0; p < N; p += 1)
			if (d_arrays[i * N + p] != d_arrays[j * N + p])
			{
				d_distances[i * M + j]++;
			}
	}
}

static int m_w = 17358;
static int m_z = 341;

__host__ unsigned int simplerand(void) {
	m_z = 36969 * (m_z & 65535) + (m_z >> 16);
	m_w = 18000 * (m_w & 65535) + (m_w >> 16);
	return (m_z << 16) + m_w;
}



__host__ void ShowGpuResults(bool *distances, bool *bitArrays)
{
	int pairCount = 0;
	HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
	cout << "All pairs:\n";
	for (int i = 0; i < M; i++)
		for (int j = 0; j < M; j++)
		{
			if (distances[i * M + j])
			{
				if (pairCount++ > 0)
					cout << ", ";
				cout << "(" << i << ", " << j << ")";
			}
		}
	cout << "\nPair count: " << pairCount << "\n";

	if (SHOW_DIFFS)
	{
		for (int i = 0; i < M; i++)
			for (int j = 0; j < M; j++)
			{
				if (distances[i * M + j])
				{
					cout << "Pair: (" << i << ", " << j << ")\n";

					for (int ind = 0; ind < N; ind++)
					{
						if (bitArrays[i * N + ind] == bitArrays[j * N + ind])
							cout << bitArrays[i * N + ind];
						else
						{
							SetConsoleTextAttribute(hConsole, 12);
							cout << bitArrays[i * N + ind];
							SetConsoleTextAttribute(hConsole, 7);
						}
					}
					cout << "\n";
					for (int ind = 0; ind < N; ind++)
					{
						if (bitArrays[i * N + ind] == bitArrays[j * N + ind])
							cout << bitArrays[j * N + ind];
						else
						{
							SetConsoleTextAttribute(hConsole, 12);
							cout << bitArrays[j * N + ind];
							SetConsoleTextAttribute(hConsole, 7);
						}
					}
					cout << "\n";
				}
			}
	}
}


__host__ void InitializeArrays(bool* arrays, bool randomDistance = false)
{
	std::minstd_rand gen(std::random_device{}());
	std::uniform_real_distribution<double> dist(0, 1);
	if (randomDistance)
		for (long long i = 0; i < M; ++i)
			for (long long j = 0; j < N; ++j)
				arrays[i * N + j] = (((double)simplerand()) / UINT_MAX) > 0.5;
	else
	{
		int* indexes = (int*)calloc(M, sizeof(int));
		int ind = 1;
		while (ind < M)
		{
			int rand = (int)((((double)simplerand()) / UINT_MAX) * N);
			indexes[ind++] = rand;
		}
		for (int i = 0; i < M; i++)
			arrays[i * N + indexes[i]] = true;

		free(indexes);
	}
}

__host__ bool CpuHammingDistance2d(bool *bitArrays)
{
	bool returnFlag = true;
	auto start = chrono::high_resolution_clock::now();
	int* distances = (int*)calloc(M * M, sizeof(int*));
	for (int i = 0; i < M; i++)
		for (int j = 0; j < M; j++)
			for (int p = 0; p < N; p++)
				if (bitArrays[i * N + p] != bitArrays[j * N + p])
					distances[i * M + j]++;
	for (int i = 0; i < M; i++)
		for (int j = 0; j < M; j++)
		{
			if (i == j)
				break;
			if (distances[i * M + j] != 1)
				returnFlag = false;
		}
	free(distances);
	auto finish = chrono::high_resolution_clock::now();
	auto miliseconds = chrono::duration_cast<chrono::milliseconds>(finish - start);
	cout << ">>>>>>CpuHammingDistance2d time: " << miliseconds.count() << " milliseconds\n";
	cout << "Result: " << returnFlag << "\n";
	return returnFlag;
}

__host__ bool GpuHammingDistance2d(bool *bitArrays)
{
	bool returnFlag = true;
	bool *d_bitArrays;
	bool *d_distances, *distances = (bool*)malloc(M * M * sizeof(bool));
	long threadCount = 1024;
	long blockCount = 1024;
	CUDA_CALL(cudaSetDevice(0));
	//size_t free = 0, total = 0;
	//cudaMemGetInfo(&free, &total);
	CUDA_CALL(cudaMalloc((void**)&d_bitArrays, M * N * sizeof(bool)));
	CUDA_CALL(cudaMalloc((void**)&d_distances, M * M * sizeof(bool)));
	CUDA_CALL(cudaMemset(d_distances, 0, M * M));
	CUDA_CALL(cudaMemcpy(d_bitArrays, bitArrays, M * N * sizeof(bool), cudaMemcpyHostToDevice));

	auto start = chrono::high_resolution_clock::now();
	CUDA_CALL(cudaDeviceSynchronize());
	cudaHammingDistance2dEquals1 << <threadCount, blockCount >> > (d_bitArrays, d_distances);
	// Check for any errors launching the kernel
	CUDA_CALL(cudaPeekAtLastError());

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	CUDA_CALL(cudaDeviceSynchronize());
	CUDA_CALL(cudaMemcpy(distances, d_distances, M * M * sizeof(bool), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaFree(d_distances));
	CUDA_CALL(cudaFree(d_bitArrays));
	CUDA_CALL(cudaDeviceReset());
	ShowGpuResults(distances, bitArrays);
	free(distances);
	auto finish = chrono::high_resolution_clock::now();
	auto milliseconds = chrono::duration_cast<chrono::milliseconds>(finish - start);
	cout << ">>>>>>GpuHammingDistance2d time: " << milliseconds.count() << " milliseconds\n";
	return true;
}

__host__ int main()
{
	auto start = chrono::high_resolution_clock::now();
	//sp³aszczona dwuwymiarowa tablica
	bool* bitArrays = (bool*)calloc(M * N, sizeof(bool));
	auto finish = chrono::high_resolution_clock::now();
	auto miliseconds = chrono::duration_cast<chrono::milliseconds>(finish - start);
	cout << ">>>>>>malloc time: " << miliseconds.count() << " milliseconds\n";

	start = chrono::high_resolution_clock::now();
	InitializeArrays(bitArrays, true);
	finish = chrono::high_resolution_clock::now();
	miliseconds = chrono::duration_cast<chrono::milliseconds>(finish - start);
	cout << ">>>>>>initialize_arrays time: " << miliseconds.count() << " milliseconds\n";
	cout << "Arrays length: " << N << " number of arrays: " << M << "\n";

	CpuHammingDistance2d(bitArrays);
	GpuHammingDistance2d(bitArrays);

	free(bitArrays);
	return 0;
}
