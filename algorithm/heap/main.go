package main

import (
	"container/heap"
	"fmt"
)

// An IntHeap is a min-heap of ints.
type IntHeap [][]int

func (h IntHeap) Len() int { return len(h) }
func (h IntHeap) Less(i, j int) bool {
	if h[i][0] != h[j][0] {
		return h[i][0] < h[j][0]
	}
	return h[i][1] > h[j][1]
}

func (h IntHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }

func (h *IntHeap) Push(x interface{}) {
	*h = append(*h, x.([]int))
}

func (h *IntHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

func main() {
	h := &IntHeap{{1, 3}, {2, 7}, {4, 8}, {3, 9}, {3, 0}, {2, 5}}
	heap.Init(h)

	for h.Len() != 0 {
		fmt.Println(heap.Pop(h))
	}
}
