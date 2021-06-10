package union_v2

// 单纯保存数组的集合
type Union struct {
	p []int
}

func MakeSet(cap int) *Union {
	u := &Union{
		p: make([]int, cap),
	}

	for i := 0; i < cap; i++ {
		u.p[i] = i
	}
	return u
}

// 合并x，y所在的集合
func (u *Union) Union(x, y int) {
	px, py := u.Find(x), u.Find(y)

	if px == py {
		return
	}
	u.p[py] = px
}

func (u *Union) Find(x int) int {
	if u.p[x] == x {
		return x
	}

	u.p[x] = u.Find(u.p[x])
	return u.p[x]
}
