#include <iostream>
#include <algorithm>
#include "omp.h"
using namespace std;

#ifndef MSORT_H

// v2 is if threshold is only used upfront, which I think isn't right (because this would never happen then)
// (note this is before parallelizing)
int* msort_recursive_v2(int* arr, size_t n) {
	if (n <= 1) {
		// no more recursion; we can back up now
		return arr;
	}
	else {
		// split in half and recursive call both halves, after smush them back together
		int half = n / 2;
		bool uneven = n % 2 == 1;

		int* left = msort_recursive_v2(arr, half);
		int* right;
		if (uneven) right = msort_recursive_v2(&arr[half], half + 1);
		else right = msort_recursive_v2(&arr[half], half);

		int* sorted;
		sorted = (int*)malloc(sizeof(int) * n);

		// merge time
		int l = 0, r = 0;
		for (int i = 0; i < n; i++) {
			if (l < half && ((r < half + 1 && uneven) || (r < half && !uneven))) {
				if (left[l] < right[r]) {
					sorted[i] = left[l];
					l++;
				}
				else {
					sorted[i] = right[r];
					r++;
				}
			}
			else if (l < half) {
				sorted[i] = left[l];
				l++;
			}
			else {
				sorted[i] = right[r];
				r++;
			}
		}

		for (int i = 0; i < n; i++) arr[i] = sorted[i];

		free(sorted);

		return arr;
	}
}

void msort_v2(int* arr, size_t n, size_t threshold) {
	if (n < threshold) sort(&arr[0], &arr[n]);
	else arr = msort_recursive_v2(arr, n);
}


int* msort_recursive(int* arr, size_t n, size_t threshold) {
	if (n < threshold) {
		// no more recursion; sort what's here and return it
		sort(&arr[0], &arr[n]);
		return arr;
	}
	else {
		// split in half and recursive call both halves, after smush them back together
		int half = n / 2;
		bool uneven = n % 2 == 1;

#pragma omp single
		{
			int* left;
			int* right;

#pragma omp task
			left = msort_recursive(arr, half, threshold);
#pragma omp task
			{
				if (uneven) right = msort_recursive(&arr[half], half + 1, threshold);
				else right = msort_recursive(&arr[half], half, threshold);
			}

#pragma omp taskwait
			int* sorted;
			sorted = (int*)malloc(sizeof(int) * n);

			// merge time
			int l = 0, r = 0;
			for (int i = 0; i < n; i++) {
				if (l < half && ((r < half + 1 && uneven) || (r < half && !uneven))) {
					if (left[l] < right[r]) {
						sorted[i] = left[l];
						l++;
					}
					else {
						sorted[i] = right[r];
						r++;
					}
				}
				else if (l < half) {
					sorted[i] = left[l];
					l++;
				}
				else {
					sorted[i] = right[r];
					r++;
				}
			}

			for (int i = 0; i < n; i++) arr[i] = sorted[i];

			free(sorted);
		}

		return arr;
	}
}

void msort(int* arr, size_t n, size_t threshold) {
	arr = msort_recursive(arr, n, threshold);
}

#endif