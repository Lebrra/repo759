#include <iostream>
using namespace std;

#ifndef SCAN_H
#define SCAN_H

class scan {
public:
	float* doScan(float* arr, int length) {
		float* scannedArr;
		scannedArr = (float*)malloc(sizeof(float) * length);

		if (scannedArr) {	// added null check here to remove warning C6011
			scannedArr[0] = arr[0];
			for (int i = 1; i < length; i++) {
				scannedArr[i] = scannedArr[i - 1] + arr[i];
			}
		}

		return scannedArr;
	}
};

#endif