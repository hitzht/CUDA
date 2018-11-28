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

#define N 80LL //rozmiar ci¹gu binarnergo
#define M 80000LL //iloœæ tablic binarnyc h
#define SHOW_DIFFS true //czy pokazywaæ ró¿nicê miêdzy kolejnymi danymi ci¹gami bitów
#define SHOW_TIME_DETAILS true //czy pokazywaæ z³o¿one komunikaty przy prezentacji czasu dzia³ania
#define PERFORMANCE_TESTS false //czy przeprowadziæ testy wydajnoœciowe na CPU i GPU

#define CUDA_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__, true); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__host__ __device__ void getTriangleArrayCoordinates(long long ind, long long &i, long long &j)
{
	i = (std::ceil(std::sqrt(2 * (ind + 1) + 0.25) - 0.5));
	j = (ind + 1 - (i - 1) * i / 2);
	j--;
	i--;
}

__global__ void cudaHammingDistance2dEquals1(bool *d_arrays, bool *d_pairs, long long arrayLength, long long numberOfArrays)
{
	const long numThreads = blockDim.x * gridDim.x;
	const long threadID = blockIdx.x * blockDim.x + threadIdx.x;
	long long i, j;
	bool flag = false;
	for (long long ind = threadID; ind < numberOfArrays * (numberOfArrays - 1) / 2 + numberOfArrays; ind += numThreads)
	{
		getTriangleArrayCoordinates(ind, i, j);
		if (i == j) continue;

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
			d_pairs[ind] = true;
	}
}

static int m_w = 17358;
static int m_z = 341;

__host__ unsigned int simplerand(void) {
	m_z = 36969 * (m_z & 65535) + (m_z >> 16);
	m_w = 18000 * (m_w & 65535) + (m_w >> 16);
	return (m_z << 16) + m_w;
}

__host__ void showCpuResults(bool *pairs, bool *bitArrays, long long arrayLength, long long numberOfArrays)
{
	long long pairCount = 0;
	cout << "All pairs:\n";
	for (long long i = 0; i < numberOfArrays; i++)
		for (long long j = 0; j < numberOfArrays; j++)
		{
			if (pairs[i * numberOfArrays + j])
			{
				if (pairCount++ > 0)
					cout << ", ";
				cout << "(" << i << ", " << j << ")";
			}
		}
	cout << "\nPair count: " << pairCount << "\n";
}

__host__ void showGpuResults(bool *pairs, bool *bitArrays, long long arrayLength, long long numberOfArrays)
{
	long pairCount = 0;
	HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
	if (SHOW_DIFFS)
	{
		for (long long ind = 0; ind < numberOfArrays * (numberOfArrays - 1) / 2 + numberOfArrays; ind++)
		{
			if (pairs[ind])
			{
				long long i, j;
				getTriangleArrayCoordinates(ind, i, j);
				cout << "Pair: (" << i << ", " << j << ")\n";

				for (long long ind2 = 0; ind2 < arrayLength; ind2++)
				{
					if (bitArrays[i * arrayLength + ind2] == bitArrays[j * arrayLength + ind2])
						cout << bitArrays[i * arrayLength + ind2];
					else
					{
						SetConsoleTextAttribute(hConsole, 12);
						cout << bitArrays[i * arrayLength + ind2];
						SetConsoleTextAttribute(hConsole, 7);
					}
				}
				cout << "\n";
				for (long long ind2 = 0; ind2 < arrayLength; ind2++)
				{
					if (bitArrays[i * arrayLength + ind2] == bitArrays[j * arrayLength + ind2])
						cout << bitArrays[j * arrayLength + ind2];
					else
					{
						SetConsoleTextAttribute(hConsole, 12);
						cout << bitArrays[j * arrayLength + ind2];
						SetConsoleTextAttribute(hConsole, 7);
					}
				}
				cout << "\n";
			}
		}
	}
	cout << "All pairs:\n";
	for (long long ind = 0; ind < numberOfArrays * (numberOfArrays - 1) / 2 + numberOfArrays; ind++)
	{
		if (pairs[ind])
		{
			long long i, j;
			getTriangleArrayCoordinates(ind, i, j);
			if (pairCount++ > 0)
				cout << ", ";
			cout << "(" << i << ", " << j << ")";
		}
	}
	cout << "\nPair count: " << pairCount << "\n";

}

__host__ void initializeArrays(bool* arrays, long long arrayLength, long long numberOfArrays, bool randomDistance = false)
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

		delete indexes;
	}
}

__host__ void cpuHammingDistance2d(bool *bitArrays, long long arrayLength, long long numberOfArrays)
{
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
	auto microseconds = chrono::duration_cast<chrono::microseconds>(finish - start);
	if (SHOW_TIME_DETAILS)
	{
		cout << ">>>>>>CpuHammingDistance2d time: " << microseconds.count() << " microseconds\n";
		showCpuResults(pairs, bitArrays, arrayLength, numberOfArrays);
	}
	else
	{
		cout << microseconds.count() << " ";
	}
	delete pairs;
}

__host__ void gpuHammingDistance2d(bool *bitArrays, long long arrayLength, long long numberOfArrays)
{
	bool *d_bitArrays;
	bool *d_pairs, *pairs = (bool*)malloc(numberOfArrays * (numberOfArrays - 1) / 2 + numberOfArrays * sizeof(bool));

	int blockSize = 1024;      // The launch configurator returned block size 
	//int minGridSize;    // The minimum grid size needed to achieve the maximum occupancy for a full device launch 
	int gridSize = 1024;       // The actual grid size needed, based on input size 

	CUDA_CALL(cudaSetDevice(0));
	//size_t free = 0, total = 0;
	//cudaMemGetInfo(&free, &total);
	auto start = chrono::high_resolution_clock::now();
	CUDA_CALL(cudaMalloc((void**)&d_bitArrays, numberOfArrays * arrayLength * sizeof(bool)));
	CUDA_CALL(cudaMalloc((void**)&d_pairs, numberOfArrays * (numberOfArrays - 1) / 2 + numberOfArrays * sizeof(bool)));
	CUDA_CALL(cudaMemset(d_pairs, 0, numberOfArrays * (numberOfArrays - 1) / 2 + numberOfArrays));
	CUDA_CALL(cudaMemcpy(d_bitArrays, bitArrays, numberOfArrays * arrayLength * sizeof(bool), cudaMemcpyHostToDevice));

	//CUDA_CALL(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cudaHammingDistance2dEquals1, 0, M * M));

	// Round up according to array size 
	//gridSize = (numberOfArrays * numberOfArrays + blockSize - 1) / blockSize;


	auto finish = chrono::high_resolution_clock::now();
	auto microseconds = chrono::duration_cast<chrono::microseconds>(finish - start);
	if (SHOW_TIME_DETAILS)
	{
		cout << ">>>>>>GpuHammingDistance2d malloc + H2D time: " << microseconds.count() << " microseconds\n";
	}
	else
	{
		cout << microseconds.count() << " ";
	}
	start = chrono::high_resolution_clock::now();
	CUDA_CALL(cudaDeviceSynchronize());
	cudaHammingDistance2dEquals1 << <gridSize, blockSize >> > (d_bitArrays, d_pairs, arrayLength, numberOfArrays);
	// Check for any errors launching the kernel
	CUDA_CALL(cudaGetLastError());

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	CUDA_CALL(cudaDeviceSynchronize());
	CUDA_CALL(cudaMemcpy(pairs, d_pairs, numberOfArrays * (numberOfArrays - 1) / 2 + numberOfArrays * sizeof(bool), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaFree(d_pairs));
	CUDA_CALL(cudaFree(d_bitArrays));
	CUDA_CALL(cudaDeviceReset());
	finish = chrono::high_resolution_clock::now();
	microseconds = chrono::duration_cast<chrono::microseconds>(finish - start);
	if (SHOW_TIME_DETAILS)
	{
		showGpuResults(pairs, bitArrays, arrayLength, numberOfArrays);
		cout << ">>>>>>GpuHammingDistance2d algorithm + D2H time: " << microseconds.count() << " microseconds\n";
	}
	else
	{
		cout << microseconds.count() << " ";
	}
	delete pairs;
}

__host__ void performanceTests()
{
	int nArray[] = { 50, 100, 250, 500, 1000, 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000, 25000, 30000, 35000, 40000, 45000, 60000, 70000, 80000, 90000, 100000, 125000, 150000, 200000, 300000, 400000, 500000, 1000000 };
	int mArray[] = { 20, 50, 100, 250, 500,	1000, 2500,	5000, 7500,	10000, 15000, 20000, 25000, 50000, 75000 };

	for (int n = 0; n != (sizeof(nArray) / sizeof(*nArray)); n++)
	{

		cout << "\nN = " << nArray[n] << "\n";
		for (int m = 0; m != (sizeof(mArray) / sizeof(*mArray)); m++)
		{
			cout << "\nM = " << mArray[m] << "\n";
			bool *bitArrays = (bool*)calloc((long long)mArray[m] * nArray[n], sizeof(bool));
			initializeArrays(bitArrays, nArray[n], mArray[m], true);

			cpuHammingDistance2d(bitArrays, nArray[n], mArray[m]);
			gpuHammingDistance2d(bitArrays, nArray[n], mArray[m]);

			delete bitArrays; 
		}
	}
}

__host__ int main()
{
	if (PERFORMANCE_TESTS)
	{
		performanceTests();
	}
	else
	{
		long long arrayLength = N;
		long long numberOfArrays = M;
		auto start = chrono::high_resolution_clock::now();
		//sp³aszczona dwuwymiarowa tablica
		bool *bitArrays = (bool*)calloc(numberOfArrays * arrayLength, sizeof(bool));
		auto finish = chrono::high_resolution_clock::now();
		auto miliseconds = chrono::duration_cast<chrono::microseconds>(finish - start);
		cout << ">>>>>>malloc time: " << miliseconds.count() << " microseconds\n";

		start = chrono::high_resolution_clock::now();
		initializeArrays(bitArrays, arrayLength, numberOfArrays, true);
		finish = chrono::high_resolution_clock::now();
		miliseconds = chrono::duration_cast<chrono::microseconds>(finish - start);
		cout << ">>>>>>initialize_arrays time: " << miliseconds.count() << " microseconds\n";
		cout << "Arrays length: " << N << " number of arrays: " << numberOfArrays << "\n";

		cpuHammingDistance2d(bitArrays, arrayLength, numberOfArrays);
		gpuHammingDistance2d(bitArrays, arrayLength, numberOfArrays);

		delete bitArrays;
	}
	return 0;
}
