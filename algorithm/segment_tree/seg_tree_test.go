package segtree_test

import (
	"math/rand"
	"testing"

	segtree "github.com/fudute/go-leetcode/algorithm/segment_tree"
)

var seg *segtree.SegTree

const N = 1000000

func init() {
	nums := make([]int, N)
	for i := 0; i < len(nums); i++ {
		nums[i] = rand.Int()
	}
	seg = segtree.NewSegTree(nums)
}

func TestSegTree(t *testing.T) {
	nums := []int{1, 2, 3, 4, 5}
	seg := segtree.NewSegTree(nums)
	if seg.RangeSum(1, 4) != 14 {
		t.Fail()
	}

	seg.UpdateAt(1, 3)
	if seg.RangeSum(1, 3) != 10 {
		t.Fail()
	}
}

// 可以确定，算法的时间复度为log2(n)
func BenchmarkDemo(b *testing.B) {
	for i := 0; i < b.N; i++ {
		l := rand.Int() % N
		r := rand.Int() % N
		if l > r {
			l, r = r, l
		}
		seg.RangeSum(l, r)
	}
}
