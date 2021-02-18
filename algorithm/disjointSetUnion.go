package algorithm

// disjointSetNode 算法 disjoint-set Union，是并查集的实现,带路径压缩和按秩合并策略
type disjointSetNode struct {
	data int // 这里可以用interface{}
	p    *disjointSetNode
	rank int
}

func makeSet(x *disjointSetNode) {
	x.p = x
	x.rank = 0
}

// 我们手里有的应该是所有的元素，而不是所有不相交的集合，所以更合理的做法应该是在union中传入两个element，
// 如果两个元素属于同一个集合，那么什么都不做
func union(x, y *disjointSetNode) {
	if x.rank > y.rank {
		y.p = x
	} else {
		x.p = y
		if x.rank == y.rank {
			y.rank++
		}
	}
}

// 返回当前集合的代表
func findSet(x *disjointSetNode) *disjointSetNode {
	if x.p != x {
		x.p = findSet(x.p)
	}
	return x.p
}

func sameComponent(x, y *disjointSetNode) bool {
	if findSet(x) == findSet(y) {
		return true
	}
	return false
}
