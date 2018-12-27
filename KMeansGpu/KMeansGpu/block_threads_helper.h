#pragma once

void calculate_blocks_threads(int &blocks, int &threads, int points_per_thread, int max_blocks, int max_threads, int points_count) {
	blocks = points_count / (max_threads * points_per_thread) + 1;
	threads = points_count / points_per_thread + (points_count % points_per_thread == 0 ? 0 : 1);
}