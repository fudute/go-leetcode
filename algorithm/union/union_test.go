package algorithm

import (
	"testing"
)

func TestUnion(t *testing.T) {
	nums := []int{1, 2, 3, 4, 5}

	for i := 0; i < len(nums); i++ {
		MakeSet(nums[i])
	}

	Union(1, 2)
	Union(3, 4)
	Union(1, 4)

	if FindSet(1) != FindSet(3) {
		t.Fail()
	}
}
