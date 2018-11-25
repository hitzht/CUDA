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

using namespace std;

#define N 1000000LL //rozmiar ci¹gu binarnergo
#define M 100LL //iloœæ tablic binarnych
#define DIST 2 //szukany dystans

#define CUDA_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

static int m_w = 17358;
static int m_z = 341;

__host__ unsigned int simplerand(void) {
	m_z = 36969 * (m_z & 65535) + (m_z >> 16);
	m_w = 18000 * (m_w & 65535) + (m_w >> 16);
	return (m_z << 16) + m_w;
}

__host__ void initialize_arrays(bool** arrays, long long const arrayLength, long long const numberOfArrays, bool randomDistance = false)
{
	std::minstd_rand gen(std::random_device{}());
	std::uniform_real_distribution<double> dist(0, 1);
	if (randomDistance)
		for (long long i = 0; i < numberOfArrays; ++i)
			for (long long j = 0; j < arrayLength; ++j)
				arrays[i][j] = (((double)simplerand() + UINT_MAX) / UINT_MAX / 2) > 0.5;
	else
	{
		int* indexes = (int*)calloc(numberOfArrays, sizeof(int));
		int ind = 1;
		while (ind < numberOfArrays)
		{
			bool flag = 0;
			int rand = (int)((((double)simplerand() + UINT_MAX) / UINT_MAX / 2) * arrayLength);
			for (int i = 0; i < numberOfArrays; i++)
				if (indexes[i] == rand)
				{
					flag = 1;
					break;
				}
			if (flag)
				continue;
			indexes[ind++] = rand;
		}
		for (int i = 0; i < numberOfArrays; i++)
			arrays[i][indexes[i]] = true;
		free(indexes);
	}
}

__host__ bool CpuHammingDistance2d(bool **bitArrays, const long long arrayLength, const long long numberOfArray, const long long distance)
{
	bool returnFlag = true;
	auto start = chrono::high_resolution_clock::now();
	int** distances = (int**)calloc(numberOfArray, sizeof(int*));
	for (int i = 0; i < numberOfArray; i++)
		distances[i] = (int*)calloc(numberOfArray, sizeof(int));
	for (int i = 0; i < numberOfArray; i++)
		for (int j = 0; j < numberOfArray; j++)
		{
			if (i == j)
				break;
			for (int p = 0; p < arrayLength; p++)
				if (bitArrays[i][p] != bitArrays[j][p])
					distances[i][j]++;
		}
	for (int i = 0; i < numberOfArray; i++)
		for (int j = 0; j < numberOfArray; j++)
		{
			if (i == j)
				break;
			if (distances[i][j] != distance)
				returnFlag = false;
		}
	for (int i = 0; i < numberOfArray; i++)
		free(distances[i]);
	free(distances);
	auto finish = chrono::high_resolution_clock::now();
	auto miliseconds = chrono::duration_cast<chrono::milliseconds>(finish - start);
	cout << "CpuHammingDistance2d time: " << miliseconds.count() << " milliseconds\n";
	cout << "Result: " << returnFlag << "\n";
	return returnFlag;
}

__host__ bool GpuHammingDistance2d(bool **bitArrays, const long long arrayLength, const long long numberOfArray, const long long distance)
{
	return true;
}

__host__ int main()
{
	long long arrayLength = N;
	long long numberOfArrays = M;
	long long distance = DIST;

	auto start = chrono::high_resolution_clock::now();
	bool** bitArrays = (bool**)malloc(numberOfArrays * sizeof(bool*));
	for (int i = 0; i < numberOfArrays; i++)
		bitArrays[i] = (bool*)calloc(arrayLength, sizeof(bool));
	auto finish = chrono::high_resolution_clock::now();
	auto miliseconds = chrono::duration_cast<chrono::milliseconds>(finish - start);
	cout << "Malloc time: " << miliseconds.count() << " milliseconds\n";

	start = chrono::high_resolution_clock::now();
	initialize_arrays(bitArrays, arrayLength, numberOfArrays, false);
	finish = chrono::high_resolution_clock::now();
	miliseconds = chrono::duration_cast<chrono::milliseconds>(finish - start);
	cout << "initialize_arrays time: " << miliseconds.count() << " milliseconds\n";
	cout << "Arrays length: " << arrayLength << " number of arrays: " << numberOfArrays << "\n";

	CpuHammingDistance2d(bitArrays, arrayLength, numberOfArrays, distance);
	GpuHammingDistance2d(bitArrays, arrayLength, numberOfArrays, distance);

	for (int i = 0; i < numberOfArrays; i++)
		free(bitArrays[i]);
	free(bitArrays);
	return 0;
}
