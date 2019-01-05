#pragma once

void calculate_blocks_threads(int &blocks, int &threads, int points_per_thread, int max_blocks, int max_threads, int points_count) {
	threads = 512;
	blocks = points_count/(threads*points_per_thread) + 1;
}