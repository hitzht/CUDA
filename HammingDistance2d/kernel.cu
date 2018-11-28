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

#define N 50LL //rozmiar ci�gu binarnergo
#define M 100000LL //ilo�� tablic binarnyc h
#define SHOW_DIFFS false //czy pokazywa� r�nic� mi�dzy kolejnymi danymi ci�gami bit�w

#define CUDA_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ void cudaHammingDistance2dEquals1(bool *d_arrays, bool *d_pairs)
{
	const long numThreads = blockDim.x * gridDim.x;
	const long threadID = blockIdx.x * blockDim.x + threadIdx.x;
	long long i, j;
	bool flag = false;
	for (long long ind = threadID; ind < M * M; ind += numThreads)
	{
		i = ind % M;
		j = ind / M;
		if (i == j)
			continue;
		if (j > i)
			continue;
		for (long long p = 0; p < N; p += 1)
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
	for (long long ind = threadID; ind < M * M; ind += numThreads)
	{
		long long i = ind % M;
		long long j = ind / M;
		if (i == j)
			continue;
		for (long long p = 0; p < N; p += 1)
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

__host__ void ShowCpuResults(bool *distances, bool *bitArrays)
{
	long long pairCount = 0;
	cout << "All pairs:\n";
	for (long long i = 0; i < M; i++)
		for (long long j = 0; j < M; j++)
		{
			if (distances[i * M + j])
			{
				if (pairCount++ > 0)
					cout << ", ";
				cout << "(" << i << ", " << j << ")";
			}
		}
	cout << "\nPair count: " << pairCount << "\n";
}

__host__ void ShowGpuResults(bool *distances, bool *bitArrays)
{
	long pairCount = 0;
	HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
	if (SHOW_DIFFS)
	{
		for (long long i = 0; i < M; i++)
			for (long long j = 0; j < M; j++)
			{
				if (distances[i * M + j])
				{
					cout << "Pair: (" << i << ", " << j << ")\n";

					for (long long ind = 0; ind < N; ind++)
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
					for (long long ind = 0; ind < N; ind++)
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
	cout << "All pairs:\n";
	for (long long i = 0; i < M; i++)
		for (long long j = 0; j < M; j++)
		{
			if (distances[i * M + j])
			{
				if (pairCount++ > 0)
					cout << ", ";
				cout << "(" << i << ", " << j << ")";
			}
		}
	cout << "\nPair count: " << pairCount << "\n";

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

__host__ void CpuHammingDistance2d(bool *bitArrays)
{
	bool returnFlag = true;
	auto start = chrono::high_resolution_clock::now();
	bool* pairs = (bool*)calloc(M * M, sizeof(bool));
	for (long long i = 0; i < M; i++)
		for (long long j = 0; j < M; j++)
		{
			if (j > i)
				break;
			bool flag = false;
			for (long long p = 0; p < N; p++)
				if (bitArrays[i * N + p] != bitArrays[j * N + p])
					if (!flag)
						flag = true;
					else
					{
						flag = false;
						break;
					}
			if (flag)
				pairs[i * M + j] = true;
		}

	auto finish = chrono::high_resolution_clock::now();
	auto miliseconds = chrono::duration_cast<chrono::milliseconds>(finish - start);
	cout << ">>>>>>CpuHammingDistance2d time: " << miliseconds.count() << " milliseconds\n";
	ShowCpuResults(pairs, bitArrays);
	free(pairs);
}

__host__ void GpuHammingDistance2d(bool *bitArrays)
{
	bool returnFlag = true;
	bool *d_bitArrays;
	bool *d_pairs, *pairs = (bool*)malloc(M * M * sizeof(bool));
	
	int blockSize = 1024;      // The launch configurator returned block size 
	//int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch 
	int gridSize = 1024;       // The actual grid size needed, based on input size 

	CUDA_CALL(cudaSetDevice(0));
	//size_t free = 0, total = 0;
	//cudaMemGetInfo(&free, &total);
	auto start = chrono::high_resolution_clock::now();
	CUDA_CALL(cudaMalloc((void**)&d_bitArrays, M * N * sizeof(bool)));
	CUDA_CALL(cudaMalloc((void**)&d_pairs, M * M * sizeof(bool)));
	CUDA_CALL(cudaMemset(d_pairs, 0, M * M));
	CUDA_CALL(cudaMemcpy(d_bitArrays, bitArrays, M * N * sizeof(bool), cudaMemcpyHostToDevice));

	//CUDA_CALL(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cudaHammingDistance2dEquals1, 0, M * M));

	// Round up according to array size 
	//gridSize = (M * M + blockSize - 1) / blockSize;


	auto finish = chrono::high_resolution_clock::now();
	auto milliseconds = chrono::duration_cast<chrono::milliseconds>(finish - start);
	cout << ">>>>>>GpuHammingDistance2d malloc + H2D time: " << milliseconds.count() << " milliseconds\n";
	start = chrono::high_resolution_clock::now();
	CUDA_CALL(cudaDeviceSynchronize());
	cudaHammingDistance2dEquals1 << <gridSize, blockSize>> > (d_bitArrays, d_pairs);
	// Check for any errors launching the kernel
	CUDA_CALL(cudaPeekAtLastError());

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	CUDA_CALL(cudaDeviceSynchronize());
	CUDA_CALL(cudaMemcpy(pairs, d_pairs, M * M * sizeof(bool), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaFree(d_pairs));
	CUDA_CALL(cudaFree(d_bitArrays));
	CUDA_CALL(cudaDeviceReset());
	finish = chrono::high_resolution_clock::now();
	milliseconds = chrono::duration_cast<chrono::milliseconds>(finish - start);
	ShowGpuResults(pairs, bitArrays);
	free(pairs);
	cout << ">>>>>>GpuHammingDistance2d algorithm + D2H time: " << milliseconds.count() << " milliseconds\n";
}

__host__ int main()
{
	auto start = chrono::high_resolution_clock::now();
	//sp�aszczona dwuwymiarowa tablica
	bool *bitArrays = (bool*)calloc(M * N, sizeof(bool));
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
