#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <string>
#include "csv_helper.h"
#include "block_threads_helper.h"
#include "math_helpers.h"
#include <stdlib.h>
#include <time.h>

using namespace std;

__device__ float ERR = 0.0001;

__global__ void kmeans(float *points, int k, float *centroids, int points_per_thread, int *points_cluster, int *n) {
	int index = threadIdx.x;
	int block = blockIdx.x;
	int start_index = index * points_per_thread * 3 + block * points_per_thread * 512 * 3,
		end_index = block * points_per_thread * 512 * 3 + 3 * index * points_per_thread + 3 * points_per_thread - 1;
	if (start_index > *n * 3) return;
	float distance, min_distance;

	for (int i = start_index; i < end_index; i += 3) {
		min_distance = FLT_MAX;
		for (int j = 0; j < k; j++) {
			distance = calculate_distance(points[i], points[i + 1], points[i + 2], centroids[j * 3], centroids[j * 3 + 1], centroids[j * 3 + 2]);
			if (distance < min_distance) {
				min_distance = distance;
				points_cluster[i / 3] = j;
			}
		}
	}
}

__global__ void calculate_centroids(int *is_centroid_stable, float *points, int *points_cluster, float *centroids, int *n) {
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

	if (fabsf(x / points_count - centroids[index * 3]) < ERR && fabsf(y / points_count - centroids[index * 3 + 1]) < ERR && fabsf(z / points_count - centroids[index * 3 + 2]) < ERR) {
		is_centroid_stable[index] = 1;
	}
	else {
		is_centroid_stable[index] = 0;
		centroids[index * 3] = x / points_count;
		centroids[index * 3 + 1] = y / points_count;
		centroids[index * 3 + 2] = z / points_count;
	}
}

int main() {
	srand(time(NULL));
	int n, threads_count, used_device_blocks, points_per_thread = 2;
	int k, is_finished = 0, max_iterations = 100, iterations = 0;

	printf("K: ");
	scanf("%d", &k);

	float *points = read_csv("D:\\Projects\\gpu\\KMeansGPU\\data1.txt", n);
	float *device_points;
	int *points_cluster, *points_cluster_device, *dev_is_centroid_stable, *dev_n, *is_centroid_stable = new int[k];
	float *centroids = new float[k*n];
	float *dev_centroids, max_x = FLT_MIN, min_x = FLT_MAX, max_y = FLT_MIN, min_y = FLT_MAX, max_z = FLT_MIN, min_z = FLT_MAX;

	for (int i = 0; i < n; i++) {
		if (points[i * 3] > max_x) {
			max_x = points[i];
		}
		if (points[i * 3] < min_x) {
			min_x = points[i];
		}
		if (points[i * 3 + 1] > max_y) {
			max_y = points[i + 1];
		}
		if (points[i * 3 + 1] < min_y) {
			min_y = points[i + 1];
		}
		if (points[i * 3 + 2] > max_z) {
			max_z = points[i + 2];
		}
		if (points[i * 3 + 2] < min_z) {
			min_z = points[i + 2];
		}
	}

	for (int i = 0; i < k; i++) {
		centroids[i * 3] = rand() % (int)max_x + min_x;
		centroids[i * 3 + 1] = rand() % (int)max_y + min_y;
		centroids[i * 3 + 2] = rand() % (int)max_z + min_z;
	}

	calculate_blocks_threads(used_device_blocks, threads_count, points_per_thread, 4096, 512, n);

	cudaError_t cudaStatus = cudaSetDevice(0);
	points_cluster = new int[n];

	for (int i = 0; i < n; i++) {
		points_cluster[i] = -1;
	}

	for (int i = 0; i < k; i++) {
		is_centroid_stable[i] = 0;
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

	cudaStatus = cudaMalloc((void**)&dev_is_centroid_stable, k * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return 0;
	}

	cudaStatus = cudaMemcpy(dev_is_centroid_stable, is_centroid_stable, k * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
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

	clock_t begin = clock();

	while (!is_finished && iterations < max_iterations) {
		kmeans <<<used_device_blocks, threads_count >>> (device_points, k, dev_centroids, points_per_thread, points_cluster_device, dev_n);
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after kmeans!\n", cudaStatus);
			return 0;
		}
		calculate_centroids<<<1, k>>>(dev_is_centroid_stable, device_points, points_cluster_device, dev_centroids, dev_n);
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after calculate_centroids!\n", cudaStatus);
			return 0;
		}
		cudaStatus = cudaMemcpy(is_centroid_stable, dev_is_centroid_stable, k * sizeof(int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy from device failed!");
			return 0;
		}
		is_finished = 1;
		for (int i = 0; i < k; i++) {
			is_finished &= is_centroid_stable[i];
		}

		iterations++;

		if (iterations % 10 == 0) {
			printf("|");
		}
	}

	printf("\n");
	clock_t end = clock();

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
	
	write_csv("out.txt", n, points, points_cluster);

	if (iterations == max_iterations) {
		printf("The method didn't converge\n");
	}
	else {
		printf("The method converge after %d iterations\n", iterations);
	}

	printf("%llu seconds ellapsed\n", uint64_t(end - begin) / CLOCKS_PER_SEC);
	printf("Results saved to out.txt\n");

	return 0;
}
