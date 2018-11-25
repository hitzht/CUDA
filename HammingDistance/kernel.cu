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

#define N 1600000000LL //rozmiar s³ów dla których liczona jest odleg³oœæ hamminga

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

__global__ void cudaHammingDistance(const char* w1, const char* w2, const long long* length)
{
	const long numThreads = blockDim.x * gridDim.x;
	const long threadID = blockIdx.x * blockDim.x + threadIdx.x;

	for (long i = threadID; i < (*length); i += numThreads)
		if (w1[i] != w2[i])
			atomicAdd(&counter, 1);
}

__host__ long long hammingDistance(char ** strings, const long length)
{
	long distance = 0;
	for (long i = 0; i < length; i++)
		if (strings[0][i] != strings[1][i])
			distance++;
	return distance;
}


static int m_w = 17358;
static int m_z = 341;

unsigned int simplerand(void) {
	m_z = 36969 * (m_z & 65535) + (m_z >> 16);
	m_w = 18000 * (m_w & 65535) + (m_w >> 16);
	return (m_z << 16) + m_w;
}

__host__ void gen_random_words(char* s1, char* s2, long long len)
{
	char alphanum[] =
		"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
	std::minstd_rand gen(std::random_device{}());
	std::uniform_real_distribution<double> dist(0, 1);
	for (long long i = 0; i < len; ++i) {
		s1[i] = alphanum[(int)((((double)simplerand() + UINT_MAX) / UINT_MAX / 2) * 62)];
		s2[i] = alphanum[(int)((((double)simplerand() + UINT_MAX) / UINT_MAX / 2) * 62)];
	}
	s1[len - 1] = '\0';
	s2[len - 1] = '\0';
}

__host__ long long CpuHammingDistance(char **strings, const long long length)
{
	auto start = chrono::high_resolution_clock::now();
	long long distance = hammingDistance(strings, length);
	auto finish = chrono::high_resolution_clock::now();
	auto microseconds = chrono::duration_cast<chrono::microseconds>(finish - start);
	cout << "Distance: " << distance << "\n";
	cout << "CPU time: " << microseconds.count() << " microseconds\n";
	return microseconds.count();
}

__host__ long long GpuHammingDistance(char **strings, const long long length)
{
	long long sum = 0;
	long threadCount = 1024;
	long blockCount = 1024;

	long long *len;
	char *d_w1, *d_w2;

	CUDA_CALL(cudaSetDevice(0));
	CUDA_CALL(cudaDeviceSynchronize());
	CUDA_CALL(cudaThreadSynchronize());
	size_t free = 0, total = 0;
	cudaMemGetInfo(&free, &total);
	CUDA_CALL(cudaMalloc((void**)&d_w1, length * sizeof(char)));
	CUDA_CALL(cudaMalloc((void**)&d_w2, length * sizeof(char)));
	CUDA_CALL(cudaMalloc((void**)&len, sizeof(long long)));
	CUDA_CALL(cudaMemcpy(d_w1, strings[0], length * sizeof(char), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(d_w2, strings[1], length * sizeof(char), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(len, &length, sizeof(long long), cudaMemcpyHostToDevice));

	auto start = chrono::high_resolution_clock::now();
	cudaHammingDistance << <blockCount, threadCount >> > (d_w1, d_w2, len);

	// Check for any errors launching the kernel
	CUDA_CALL(cudaGetLastError());

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	CUDA_CALL(cudaDeviceSynchronize());
	CUDA_CALL(cudaMemcpyFromSymbol(&sum, counter, sizeof(int)));

	CUDA_CALL(cudaFree(d_w1));
	CUDA_CALL(cudaFree(d_w2));
	CUDA_CALL(cudaDeviceReset());

	auto finish = chrono::high_resolution_clock::now();
	auto microseconds = chrono::duration_cast<chrono::microseconds>(finish - start);
	cout << "Distance: " << sum << "\n";
	cout << "GPU time: " << microseconds.count() << " microseconds\n";
	return microseconds.count();
}

__host__ int main()
{
	long long length = N;
	char *string[2];
	auto start = chrono::high_resolution_clock::now();
	string[0] = (char*)malloc(length * sizeof(char));
	string[1] = (char*)malloc(length * sizeof(char));
	auto finish = chrono::high_resolution_clock::now();
	auto microseconds = chrono::duration_cast<chrono::microseconds>(finish - start);
	cout << "Malloc time: " << microseconds.count() << " microseconds\n";

	start = chrono::high_resolution_clock::now();
	gen_random_words(string[0], string[1], length);
	finish = chrono::high_resolution_clock::now();
	microseconds = chrono::duration_cast<chrono::microseconds>(finish - start);
	cout << "gen_random time: " << microseconds.count() << " microseconds\n";
	cout << "Length: " << length << "\n";

	CpuHammingDistance(string, length);
	GpuHammingDistance(string, length);
	free(string[0]);
	free(string[1]);
	return 0;
}
