#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <string>
#include "csv_helper.h"
#include "block_threads_helper.h"

using namespace std;

__global__ void kmeans(float *points, int k, int *centroids, int points_per_thread) {
	int index = threadIdx.x;
	int block = blockIdx.x;
	int start_index = index * points_per_thread, end_index = index * points_per_thread + points_per_thread - 1;

	printf("%d, %d - [%d, %d]\n", index, block, start_index, end_index);

}

int main() {
	
	int n, threads_count, used_device_blocks, points_per_thread = 3;
	int k = 2;
	float *points = read_csv("D:\\Projects\\gpu\\KMeansGPU\\test1.txt", n);
	float *device_points;

	int *centroids = new int[k];
	int *dev_centroids;
	centroids[0] = 0;
	centroids[1] = 1;

	calculate_blocks_threads(used_device_blocks, threads_count, points_per_thread, 4096, 512, n);

	cudaError_t cudaStatus = cudaSetDevice(0);

	cudaStatus = cudaMalloc((void**)&device_points, n * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return 0;
	}

	cudaStatus = cudaMemcpy(device_points, points, n * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return 0;
	}

	cudaStatus = cudaMalloc((void**)&dev_centroids, k * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return 0;
	}

	cudaStatus = cudaMemcpy(dev_centroids, centroids, k * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return 0;
	}

	kmeans<<<used_device_blocks, threads_count>>>(device_points, k, centroids, points_per_thread);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching decrypt!\n", cudaStatus);
		return 0;
	}

    return 0;
}
