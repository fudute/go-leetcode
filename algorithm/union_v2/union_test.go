package union_v2_test

import (
	"testing"

	"github.com/fudute/go-leetcode/algorithm/union_v2"
)

func TestUnion_V2(t *testing.T) {
	u := union_v2.MakeSet(5)

	u.Union(1, 2)
	u.Union(3, 4)

	if u.Find(1) == u.Find(3) {
		t.Fail()
	}

	u.Union(2, 4)
	if u.Find(1) != u.Find(3) {
		t.Fail()
	}
}
