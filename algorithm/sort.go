package algorithm

import "math"

// SortByValue 升序排列
type SortByValue []int

func (a SortByValue) Len() int           { return len(a) }
func (a SortByValue) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a SortByValue) Less(i, j int) bool { return a[i] < a[j] }

//QuickSort 快速排序
func QuickSort(A []int) {
	if len(A) <= 1 {
		return
	}
	cmp := A[0]
	low, high := 0, len(A)-1
	for low < high {
		for low < high && A[high] >= cmp {
			high--
		}
		A[low] = A[high]
		for low < high && A[low] <= cmp {
			low++
		}
		A[high] = A[low]
	}
	A[low] = cmp
	QuickSort(A[0:low])
	QuickSort(A[low+1:])
	return
}

// MergeSort 归并排序
func MergeSort(A []int) []int {
	if len(A) < 2 {
		return A
	}
	if len(A) == 2 {
		if A[0] > A[1] {
			A[0], A[1] = A[1], A[0]
		}
		return A
	}
	mid := len(A) / 2
	return merge(MergeSort(A[0:mid]), MergeSort(A[mid:]))
}

func merge(A, B []int) []int {
	C := make([]int, len(A)+len(B))

	var ind, p, q int
	for ind < len(C) {
		a := getOrDefault(A, p, math.MaxInt32)
		b := getOrDefault(B, q, math.MaxInt32)
		if a < b {
			C[ind] = a
			p++
		} else {
			C[ind] = b
			q++
		}
		ind++
	}
	return C
}

func getOrDefault(A []int, ind, defaultVal int) int {
	if ind < len(A) {
		return A[ind]
	}
	return defaultVal
}
