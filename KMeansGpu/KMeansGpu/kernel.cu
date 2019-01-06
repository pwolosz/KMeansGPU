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
#include <float.h>

using namespace std;

__device__ float ERR = 0.0001;

int THREADS = 512;

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

__global__ void calculate_tmp_centroids(float *tmp_centroids, int k, int threads_count, int *points_count, int *n, int *points_cluster, float* points) {
	int index = threadIdx.x;
	int block = blockIdx.x;
	int points_per_thread = *n / threads_count + 1;
	int start_index = index * points_per_thread;
	int end_index = (index+1) * points_per_thread;
	float x = 0, y = 0, z = 0;
	int points_this = 0;

	end_index = end_index > *n ? *n : end_index;

	if (start_index > *n) return;

	for (int i = start_index; i < end_index; i++) {
		if (points_cluster[i] == block) {
			x += points[i * 3];
			y += points[i * 3 + 1];
			z += points[i * 3 + 2];
			points_this++;
		}
	}

	tmp_centroids[index * 3 + threads_count * block * 3] = x;
	tmp_centroids[index * 3 + threads_count * block * 3 + 1] = y;
	tmp_centroids[index * 3 + threads_count * block * 3 + 2] = z;
	points_count[index + threads_count * block] = points_this;
}

__global__ void calculate_centroids(int *is_centroid_stable, float *centroids, float *tmp_centroids, int *points_count, int threads_count) {
	int index = threadIdx.x;
	float x = 0, y = 0, z = 0;
	int points_this = 0;
	int start_index = index * threads_count;
	int end_index = (index + 1) * threads_count;

	//printf("%d: %d - %d\n", index, start_index, end_index);

	for (int i = start_index; i < end_index; i++) {
		x += tmp_centroids[i * 3];
		y += tmp_centroids[i * 3 + 1];
		z += tmp_centroids[i * 3 + 2];
		points_this += points_count[i];
	}

	if (fabsf(x / points_this - centroids[index * 3]) < ERR && fabsf(y / points_this - centroids[index * 3 + 1]) < ERR && fabsf(z / points_this - centroids[index * 3 + 2]) < ERR) {
		is_centroid_stable[index] = 1;
	}
	else {
		centroids[3 * index] = x / points_this;
		centroids[3 * index + 1] = y / points_this;
		centroids[3 * index + 2] = z / points_this;
		is_centroid_stable[index] = 0;
	}

}

int main() {
	srand(time(NULL));
	int n, threads_count, used_device_blocks, points_per_thread = 100;
	int k, is_finished = 0, max_iterations = 100, iterations = 0;

	printf("K: ");
	scanf("%d", &k);

	float *points = read_csv("D:\\Projects\\gpu\\KMeansGPU\\data1.txt", n);
	float *device_points;
	int *points_cluster, *points_cluster_device, *dev_is_centroid_stable, *dev_n, *is_centroid_stable = new int[k];
	float *centroids = new float[k*3];
	float *dev_centroids, max_x = FLT_MIN, min_x = FLT_MAX, max_y = FLT_MIN, min_y = FLT_MAX, max_z = FLT_MIN, min_z = FLT_MAX;
	int *dev_points_count;
	float *dev_tmp_centroids;

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

	calculate_blocks_threads(used_device_blocks, threads_count, points_per_thread, 4096, THREADS, n);

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

	cudaStatus = cudaMalloc((void**)&dev_centroids, k * 3 * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return 0;
	}

	cudaStatus = cudaMemcpy(dev_centroids, centroids, k * 3 * sizeof(float), cudaMemcpyHostToDevice);
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

	cudaStatus = cudaMalloc((void**)&dev_points_count, sizeof(int) * THREADS * k);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		return 0;
	}

	cudaStatus = cudaMalloc((void**)&dev_tmp_centroids, 3 * k * THREADS * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
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

		calculate_tmp_centroids <<<k, THREADS>>>(dev_tmp_centroids, k, THREADS, dev_points_count, dev_n, points_cluster_device, device_points);
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after calculate_centroids!\n", cudaStatus);
			return 0;
		}

		calculate_centroids <<<1, k >>> (dev_is_centroid_stable, dev_centroids, dev_tmp_centroids, dev_points_count, THREADS);
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
	printf("%llu seconds ellapsed\n", uint64_t(end - begin) / CLOCKS_PER_SEC);

	write_csv("out.txt", n, points, points_cluster);

	if (iterations == max_iterations) {
		printf("The method didn't converge\n");
	}
	else {
		printf("The method converge after %d iterations\n", iterations);
	}

	printf("Results saved to out.txt\n");

	return 0;
}
