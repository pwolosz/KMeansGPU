#pragma once
#include "cuda_runtime.h"
#include <cmath>

__device__ float calculate_distance(float x1, float y1, float z1, float x2, float y2, float z2) {
	return sqrtf(powf(x2 - x1, 2) + powf(y2 - y1, 2) + powf(z2 - z1, 2));
}