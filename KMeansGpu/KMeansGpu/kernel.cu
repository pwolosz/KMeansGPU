#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <string>
#include "csv_helper.h"
#include "block_threads_helper.h"
#include "math_helpers.h"

using namespace std;

__device__ float ERR = 1e5;

__global__ void kmeans(float *points, int k, float *centroids, int points_per_thread, int *points_cluster, int *n) {
	int index = threadIdx.x;
	int block = blockIdx.x;
	int start_index = index * points_per_thread * 3, end_index = 3 * index * points_per_thread + 3 * points_per_thread - 1;
	float distance, min_distance;
	bool any_changes = false;
	printf("%d, %d - [%d, %d]\n", index, block, start_index, end_index);

	for (int i = start_index; i < end_index; i += 3) {
		min_distance = FLT_MAX;
		for (int j = 0; j < k; j++) {
			distance = calculate_distance(points[i], points[i + 1], points[i + 2], centroids[j * 3], centroids[j * 3 + 1], centroids[j * 3 + 2]);
			if (distance < min_distance) {
				min_distance = distance;

				if (points_cluster[i / 3] != j) {
					any_changes = true;
				}

				points_cluster[i / 3] = j;
			}
		}
	}
}

__global__ void calculate_centroids(int *is_finished, float *points, int *points_cluster, float *centroids, int *n) {
	*is_finished = 1;
	int index = threadIdx.x;
	int points_count = 0;
	float x = 0, y = 0, z = 0;
	for (int i = 0; i < *n; i++) {
		if (points_cluster[i] == index) {
			x += points[i * 3];
			y += points[i * 3 + 1];
			z += points[i * 3 + 2];
			points_count++;
		}
	}

	centroids[index * 3] = x / points_count;
	centroids[index * 3 + 1] = y / points_count;
	centroids[index * 3 + 2] = z / points_count;

	for (int i = 0; i < *n; i++) {
		printf("%d: %d\n", i, points_cluster[i]);
	}
}

int main() {

	int n, threads_count, used_device_blocks, points_per_thread = 3;
	int k = 2, is_finished = 0;
	float *points = read_csv("D:\\Projects\\gpu\\KMeansGPU\\test1.txt", n);
	float *device_points;
	int *points_cluster, *points_cluster_device, *dev_is_finished, *dev_n;
	float *centroids = new float[k*n];
	float *dev_centroids;
	centroids[0] = 0;
	centroids[1] = 0;
	centroids[2] = 1;
	centroids[3] = 10;
	centroids[4] = 10;
	centroids[5] = 1;
	calculate_blocks_threads(used_device_blocks, threads_count, points_per_thread, 4096, 512, n);

	cudaError_t cudaStatus = cudaSetDevice(0);
	points_cluster = new int[n];

	for (int i = 0; i < n; i++) {
		points_cluster[i] = -1;
	}

	for (int i = 0; i < n * 3; i += 3) {
		printf("%f, %f, %f\n", points[i], points[i + 1], points[i + 2]);
	}

	cudaStatus = cudaMalloc((void**)&device_points, n * 3 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return 0;
	}

	cudaStatus = cudaMemcpy(device_points, points, n * 3 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		return 0;
	}

	cudaStatus = cudaMalloc((void**)&dev_centroids, k * n * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return 0;
	}

	cudaStatus = cudaMemcpy(dev_centroids, centroids, k * n * sizeof(float), cudaMemcpyHostToDevice);
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

	cudaStatus = cudaMalloc((void**)&dev_n, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return 0;
	}

	cudaStatus = cudaMemcpy(dev_n, &n, sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy n failed!");
		return 0;
	}

	printf("n = %d\n", n);
	printf("threads = %d\n", threads_count);
	printf("blocks = %d\n", used_device_blocks);

	while (!is_finished) {
		kmeans << <used_device_blocks, threads_count >> > (device_points, k, dev_centroids, points_per_thread, points_cluster_device, dev_n);
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after kmeans!\n", cudaStatus);
			return 0;
		}
		calculate_centroids<<<1, 1>>>(dev_is_finished, device_points, points_cluster_device, dev_centroids, dev_n);
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after calculate_centroids!\n", cudaStatus);
			return 0;
		}
		cudaStatus = cudaMemcpy(&is_finished, dev_is_finished, sizeof(int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy from device failed!");
			return 0;
		}
	}

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return 0;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after kmeans!\n", cudaStatus);
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
