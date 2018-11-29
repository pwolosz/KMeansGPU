#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ int replace_and_check_centroids(int *prev_centroids, int *centroids, int k) {
	int is_finished = 1;

	for (int i = 0; i < k; i++) {
		if (prev_centroids[i] != centroids[i]) {
			is_finished = 0;
		}

		prev_centroids[i] = centroids[i];
	}

	return is_finished;
}
