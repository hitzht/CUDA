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

#define N 50LL //rozmiar ci¹gu binarnergo
#define M 50000LL //iloœæ tablic binarnyc h
#define SHOW_DIFFS false //czy pokazywaæ ró¿nicê miêdzy kolejnymi danymi ci¹gami bitów

#define CUDA_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ void cudaHammingDistance2dEquals1(bool *d_arrays, bool *d_pairs, long long arrayLength, long long numberOfArrays)
{
	const long numThreads = blockDim.x * gridDim.x;
	const long threadID = blockIdx.x * blockDim.x + threadIdx.x;
	long long i, j;
	bool flag = false;
	for (long long ind = threadID; ind < numberOfArrays * numberOfArrays; ind += numThreads)
	{
		i = ind % numberOfArrays;
		j = ind / numberOfArrays;
		if (i == j)
			continue;
		if (j > i)
			continue;
		for (long long p = 0; p < arrayLength; p += 1)
			if (d_arrays[i * arrayLength + p] != d_arrays[j * arrayLength + p])
				if (flag)
				{
					flag = false;
					break;
				}
				else
					flag = true;
		if (flag)
		{
			d_pairs[i * numberOfArrays + j] = true;
		}
	}
}

__global__ void cudaHammingDistance2d(bool *d_arrays, unsigned long long *d_distances, long long arrayLength, long long numberOfArrays)
{
	const long numThreads = blockDim.x * gridDim.x;
	const long threadID = blockIdx.x * blockDim.x + threadIdx.x;
	for (long long ind = threadID; ind < numberOfArrays * numberOfArrays; ind += numThreads)
	{
		long long i = ind % numberOfArrays;
		long long j = ind / numberOfArrays;
		if (i == j)
			continue;
		for (long long p = 0; p < arrayLength; p += 1)
			if (d_arrays[i * arrayLength + p] != d_arrays[j * arrayLength + p])
			{
				d_distances[i * numberOfArrays + j]++;
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

__host__ void ShowCpuResults(bool *distances, bool *bitArrays, long long arrayLength, long long numberOfArrays)
{
	long long pairCount = 0;
	cout << "All pairs:\n";
	for (long long i = 0; i < numberOfArrays; i++)
		for (long long j = 0; j < numberOfArrays; j++)
		{
			if (distances[i * numberOfArrays + j])
			{
				if (pairCount++ > 0)
					cout << ", ";
				cout << "(" << i << ", " << j << ")";
			}
		}
	cout << "\nPair count: " << pairCount << "\n";
}

__host__ void ShowGpuResults(bool *distances, bool *bitArrays, long long arrayLength, long long numberOfArrays)
{
	long pairCount = 0;
	HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
	if (SHOW_DIFFS)
	{
		for (long long i = 0; i < numberOfArrays; i++)
			for (long long j = 0; j < numberOfArrays; j++)
			{
				if (distances[i * numberOfArrays + j])
				{
					cout << "Pair: (" << i << ", " << j << ")\n";

					for (long long ind = 0; ind < arrayLength; ind++)
					{
						if (bitArrays[i * arrayLength + ind] == bitArrays[j * arrayLength + ind])
							cout << bitArrays[i * arrayLength + ind];
						else
						{
							SetConsoleTextAttribute(hConsole, 12);
							cout << bitArrays[i * arrayLength + ind];
							SetConsoleTextAttribute(hConsole, 7);
						}
					}
					cout << "\n";
					for (long long ind = 0; ind < arrayLength; ind++)
					{
						if (bitArrays[i * arrayLength + ind] == bitArrays[j * arrayLength + ind])
							cout << bitArrays[j * arrayLength + ind];
						else
						{
							SetConsoleTextAttribute(hConsole, 12);
							cout << bitArrays[j * arrayLength + ind];
							SetConsoleTextAttribute(hConsole, 7);
						}
					}
					cout << "\n";
				}
			}
	}
	cout << "All pairs:\n";
	for (long long i = 0; i < numberOfArrays; i++)
		for (long long j = 0; j < numberOfArrays; j++)
		{
			if (distances[i * numberOfArrays + j])
			{
				if (pairCount++ > 0)
					cout << ", ";
				cout << "(" << i << ", " << j << ")";
			}
		}
	cout << "\nPair count: " << pairCount << "\n";

}

__host__ void InitializeArrays(bool* arrays, long long arrayLength, long long numberOfArrays, bool randomDistance = false)
{
	std::minstd_rand gen(std::random_device{}());
	std::uniform_real_distribution<double> dist(0, 1);
	if (randomDistance)
		for (long long i = 0; i < numberOfArrays; ++i)
			for (long long j = 0; j < arrayLength; ++j)
				arrays[i * arrayLength + j] = (((double)simplerand()) / UINT_MAX) > 0.5;
	else
	{
		int* indexes = (int*)calloc(numberOfArrays, sizeof(int));
		int ind = 1;
		while (ind < numberOfArrays)
		{
			int rand = (int)((((double)simplerand()) / UINT_MAX) * arrayLength);
			indexes[ind++] = rand;
		}
		for (int i = 0; i < numberOfArrays; i++)
			arrays[i * arrayLength + indexes[i]] = true;

		free(indexes);
	}
}

__host__ void CpuHammingDistance2d(bool *bitArrays, long long arrayLength, long long numberOfArrays)
{
	bool returnFlag = true;
	auto start = chrono::high_resolution_clock::now();
	bool* pairs = (bool*)calloc(numberOfArrays * numberOfArrays, sizeof(bool));
	for (long long i = 0; i < numberOfArrays; i++)
		for (long long j = 0; j < numberOfArrays; j++)
		{
			if (j > i)
				break;
			bool flag = false;
			for (long long p = 0; p < arrayLength; p++)
				if (bitArrays[i * arrayLength + p] != bitArrays[j * arrayLength + p])
					if (!flag)
						flag = true;
					else
					{
						flag = false;
						break;
					}
			if (flag)
				pairs[i * numberOfArrays + j] = true;
		}

	auto finish = chrono::high_resolution_clock::now();
	auto miliseconds = chrono::duration_cast<chrono::milliseconds>(finish - start);
	cout << ">>>>>>CpuHammingDistance2d time: " << miliseconds.count() << " milliseconds\n";
	ShowCpuResults(pairs, bitArrays, arrayLength, numberOfArrays);
	free(pairs);
}

__host__ void GpuHammingDistance2d(bool *bitArrays, long long arrayLength, long long numberOfArrays)
{
	bool returnFlag = true;
	bool *d_bitArrays;
	bool *d_pairs, *pairs = (bool*)malloc(numberOfArrays * numberOfArrays * sizeof(bool));
	
	int blockSize = 1024;      // The launch configurator returned block size 
	//int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch 
	int gridSize = 1024;       // The actual grid size needed, based on input size 

	CUDA_CALL(cudaSetDevice(0));
	//size_t free = 0, total = 0;
	//cudaMemGetInfo(&free, &total);
	auto start = chrono::high_resolution_clock::now();
	CUDA_CALL(cudaMalloc((void**)&d_bitArrays, numberOfArrays * arrayLength * sizeof(bool)));
	CUDA_CALL(cudaMalloc((void**)&d_pairs, numberOfArrays * numberOfArrays * sizeof(bool)));
	CUDA_CALL(cudaMemset(d_pairs, 0, numberOfArrays * numberOfArrays));
	CUDA_CALL(cudaMemcpy(d_bitArrays, bitArrays, numberOfArrays * arrayLength * sizeof(bool), cudaMemcpyHostToDevice));

	//CUDA_CALL(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cudaHammingDistance2dEquals1, 0, M * M));

	// Round up according to array size 
	//gridSize = (numberOfArrays * numberOfArrays + blockSize - 1) / blockSize;


	auto finish = chrono::high_resolution_clock::now();
	auto milliseconds = chrono::duration_cast<chrono::milliseconds>(finish - start);
	cout << ">>>>>>GpuHammingDistance2d malloc + H2D time: " << milliseconds.count() << " milliseconds\n";
	start = chrono::high_resolution_clock::now();
	CUDA_CALL(cudaDeviceSynchronize());
	cudaHammingDistance2dEquals1 << <gridSize, blockSize>> > (d_bitArrays, d_pairs, arrayLength, numberOfArrays);
	// Check for any errors launching the kernel
	CUDA_CALL(cudaPeekAtLastError());

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	CUDA_CALL(cudaDeviceSynchronize());
	CUDA_CALL(cudaMemcpy(pairs, d_pairs, numberOfArrays * numberOfArrays * sizeof(bool), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaFree(d_pairs));
	CUDA_CALL(cudaFree(d_bitArrays));
	CUDA_CALL(cudaDeviceReset());
	finish = chrono::high_resolution_clock::now();
	milliseconds = chrono::duration_cast<chrono::milliseconds>(finish - start);
	ShowGpuResults(pairs, bitArrays, arrayLength, numberOfArrays);
	free(pairs);
	cout << ">>>>>>GpuHammingDistance2d algorithm + D2H time: " << milliseconds.count() << " milliseconds\n";
}

__host__ int main()
{
	long long arrayLength = N;
	long long numberOfArrays = M;
	auto start = chrono::high_resolution_clock::now();
	//sp³aszczona dwuwymiarowa tablica
	bool *bitArrays = (bool*)calloc(numberOfArrays * arrayLength, sizeof(bool));
	auto finish = chrono::high_resolution_clock::now();
	auto miliseconds = chrono::duration_cast<chrono::milliseconds>(finish - start);
	cout << ">>>>>>malloc time: " << miliseconds.count() << " milliseconds\n";

	start = chrono::high_resolution_clock::now();
	InitializeArrays(bitArrays, arrayLength, numberOfArrays, true);
	finish = chrono::high_resolution_clock::now();
	miliseconds = chrono::duration_cast<chrono::milliseconds>(finish - start);
	cout << ">>>>>>initialize_arrays time: " << miliseconds.count() << " milliseconds\n";
	cout << "Arrays length: " << N << " number of arrays: " << numberOfArrays << "\n";

	CpuHammingDistance2d(bitArrays, arrayLength, numberOfArrays);
	GpuHammingDistance2d(bitArrays, arrayLength, numberOfArrays);

	free(bitArrays);
	return 0;
}
