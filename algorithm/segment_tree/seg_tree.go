package segtree

// 线段树，实现了快速的区间求和
type SegTree struct {
	l, r        int
	left, right *SegTree
	sum         int
}

func NewSegTree(nums []int) *SegTree {
	if len(nums) == 1 {
		return &SegTree{}
	}

	return buildSegTree(0, len(nums)-1, nums)
}

func buildSegTree(l, r int, nums []int) *SegTree {
	seg := &SegTree{l: l, r: r}
	if l == r {
		seg.sum = nums[l]
		return seg
	}

	mid := (l + r) / 2
	seg.left = buildSegTree(l, mid, nums)
	seg.right = buildSegTree(mid+1, r, nums)

	seg.sum = seg.left.sum + seg.right.sum
	return seg
}

func (seg *SegTree) RangeSum(L, R int) int {
	return seg.rangeSumQuery(L, R)
}

func (seg *SegTree) rangeSumQuery(L, R int) int {
	if seg.l >= L && seg.r <= R {
		return seg.sum
	}
	if seg.l > R || seg.r < L {
		return 0
	}

	if L == R {
		return seg.sum
	}

	left := seg.left.rangeSumQuery(L, R)
	right := seg.right.rangeSumQuery(L, R)

	return left + right
}

func (seg *SegTree) UpdateAt(idx, val int) {
	seg.rangeSumUpdate([]int{val}, idx, idx)
}

func (seg *SegTree) UpdateRange(beg int, new_val []int) {
	seg.rangeSumUpdate(new_val, beg, beg+len(new_val))
}

func (seg *SegTree) rangeSumUpdate(new_vals []int, L, R int) int {
	if seg.l > R || seg.r < L {
		return seg.sum
	}
	if seg.l == seg.r {
		seg.sum = new_vals[seg.l-L]
		return seg.sum
	}

	left := seg.left.rangeSumUpdate(new_vals, L, R)
	right := seg.right.rangeSumUpdate(new_vals, L, R)

	seg.sum = left + right
	return seg.sum
}
