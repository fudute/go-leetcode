package main

// MapSum leetcode impl
type MapSum struct {
	sum  map[string]int
	data map[string]int
}

func Constructor() MapSum {
	ms := MapSum{
		sum:  make(map[string]int),
		data: make(map[string]int),
	}
	return ms
}

func (this *MapSum) Insert(key string, val int) {
	for i := 1; i <= len(key); i++ {
		this.sum[key[:i]] += val
	}

	// 如果之前存在key, 则减去对应的值
	if _, ok := this.data[key]; ok {
		for i := 1; i <= len(key); i++ {
			this.sum[key[:i]] -= this.data[key]
		}
	}

	this.data[key] = val
}

func (this *MapSum) Sum(prefix string) int {

	if v, ok := this.sum[prefix]; ok {
		return v
	}
	return 0
}
