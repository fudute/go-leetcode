package main

import (
	"container/list"
)

type valPair struct {
	value int
	elem  *list.Element
}

type LRUCache struct {
	lst      list.List
	m        map[int]valPair
	capacity int
}

func NewLRUCache(capacity int) LRUCache {
	lru := LRUCache{
		lst: list.List{},
		m:   make(map[int]valPair),
	}
	lru.capacity = capacity
	return lru
}

func (this *LRUCache) Get(key int) int {
	// 如果已经存在
	if v, ok := this.m[key]; ok {
		this.lst.MoveToFront(v.elem)
		return v.value
	}
	return -1
}

func (this *LRUCache) Put(key int, value int) {
	if v, ok := this.m[key]; ok {
		this.lst.MoveToFront(v.elem)
		this.m[key] = valPair{
			value: value,
			elem:  this.lst.Front(),
		}
		return
	}

	this.lst.PushFront(key)
	this.m[key] = valPair{
		value: value,
		elem:  this.lst.Front(),
	}

	if len(this.m) > this.capacity {
		k := this.lst.Back().Value.(int)
		this.lst.Remove(this.lst.Back())
		delete(this.m, k)
	}
}
