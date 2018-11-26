
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <string>

using namespace std;

__global__ void kmeans()
{

}

int main()
{
	ifstream file;
	file.open("D:\\Projects\\gpu\\KMeansGPU\\test1.txt");
	string line;
	string val;
	int n;
	float *points;

	if (file.is_open()) {
		getline(file, line);
		n = stoi(line);
		cout << n << endl;
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

	for (int i = 0; i < n*3; i++) {
		printf("%f\n", points[i]);
	}

	file.close();
    return 0;
}
