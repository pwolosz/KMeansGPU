#pragma once
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <string>

using namespace std;

float *read_csv(string name, int &n) {
	ifstream file;
	string line, val;
	float *points;

	file.open(name);

	if (file.is_open()) {
		getline(file, line);
		n = stoi(line);
		int index = 0;
		points = new float[n * 3];

		while (getline(file, line)) {
			int prev_val_index = 0;

			for (int i = 0; i < 3; i++) {
				int val_index = line.find(',', prev_val_index);
				val = line.substr(prev_val_index, val_index == -1 ? line.length() : val_index - prev_val_index);
				points[index] = stof(val);
				index += 1;

				prev_val_index = val_index + 1;
			}
		}
	}

	file.close();

	printf("CSV read\n");

	return points;
}

void write_csv(string name, int n, float* points, int *points_cluster) {
	ofstream file;

	file.open(name);

	if (file.is_open()) {
		for (int i = 0; i < n; i++) {
			file << points[i * 3] << ",";
			file << points[i * 3 + 1] << ",";
			file << points[i * 3 + 2] << ",";
			file << points_cluster[i] << "\n";
		}

		file.close();
	}
}