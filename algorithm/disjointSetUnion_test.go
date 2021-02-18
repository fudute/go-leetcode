package algorithm

import (
	"testing"
)

func Test_disjointSet(t *testing.T) {
	elems := make([]*disjointSetNode, 0)

	count := 4
	for i := 0; i < count; i++ {
		elems = append(elems, &disjointSetNode{data: i})
		makeSet(elems[i])
	}

	union(elems[0], elems[1])
	union(elems[2], elems[3])

	if !sameComponent(elems[0], elems[1]) {
		t.Errorf("union error")
	}

	if sameComponent(elems[0], elems[2]) {
		t.Errorf("union error")
	}
}
