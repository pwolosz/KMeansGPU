#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <string>
#include "csv_helper.h"
#include "block_threads_helper.h"
#include "device_helpers.h"
#include "math_helpers.h"

using namespace std;

__global__ void kmeans(float *points, int k, int *centroids, int *prev_centroids, int points_per_thread, int *points_cluster, int *is_finished) {
	int index = threadIdx.x;
	int block = blockIdx.x;
	int start_index = index * points_per_thread * 3, end_index = 3 * index * points_per_thread + 3 * points_per_thread - 1;
	
	printf("%d, %d - [%d, %d]\n", index, block, start_index, end_index);

	while (*is_finished != 1) {

		for (int i = start_index; i <= end_index; i += 3) {
			float min_dist = FLT_MAX, dist;
			int closest_centroid = -1;

			for (int j = 0; j < k; j++) {
				dist = calculate_distance(points[i], points[i + 1], points[i + 2], points[centroids[j]], points[centroids[j] + 1], points[centroids[j] + 2]);
				printf("%d, %d, %d, %f\n", index, i, j, dist);
				if (dist < min_dist) {
					min_dist = dist;
					closest_centroid = j;
				}
			}

			points_cluster[i/3] = closest_centroid;
		}

		*is_finished = replace_and_check_centroids(prev_centroids, centroids, k);
	}
}

int main() {
	
	int n, threads_count, used_device_blocks, points_per_thread = 3;
	int k = 2;
	float *points = read_csv("D:\\Projects\\gpu\\KMeansGPU\\test1.txt", n);
	float *device_points;
	int *points_cluster, *points_cluster_device, *prev_centroids, *dev_is_finished, *prev_centroids_device;

	int *centroids = new int[k];
	int *dev_centroids;
	centroids[0] = 0;
	centroids[1] = 3;

	calculate_blocks_threads(used_device_blocks, threads_count, points_per_thread, 4096, 512, n);

	cudaError_t cudaStatus = cudaSetDevice(0);
	points_cluster = new int[n];
	prev_centroids = new int[k];

	prev_centroids[0] = 0;
	prev_centroids[1] = 3;

	for (int i = 0; i < n; i++) {
		points_cluster[i] = -1;
	}

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

	cudaStatus = cudaMalloc((void**)&points_cluster_device, n * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return 0;
	}

	cudaStatus = cudaMemcpy(points_cluster_device, points_cluster, n * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return 0;
	}

	cudaStatus = cudaMalloc((void**)&dev_is_finished, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return 0;
	}

	cudaStatus = cudaMalloc((void**)&prev_centroids_device, k * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return 0;
	}

	cudaStatus = cudaMemcpy(prev_centroids_device, prev_centroids, k * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return 0;
	}

	kmeans<<<used_device_blocks, threads_count >>>(device_points, k, dev_centroids, prev_centroids_device, points_per_thread, points_cluster_device, dev_is_finished);

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

	cudaStatus = cudaMemcpy(points_cluster, points_cluster_device, n * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy from device failed!");
		return 0;
	}

	printf("-----------------\n");

	for (int i = 0; i < n; i++) {
		printf("%d - %d\n", i, points_cluster[i]);
	}

    return 0;
}
