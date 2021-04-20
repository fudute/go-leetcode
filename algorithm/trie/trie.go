package trie

import "strings"

type node struct {
	isEnd bool    // 表示当前节点是否是一个单词的末尾字符
	next  []*node // 接下来的字符，长度为26
}

// Trie 字典树的实现
type Trie struct {
	words []*node
}

// New 构造字典树
func New() *Trie {
	return &Trie{
		words: make([]*node, 26),
	}
}

// Put 添加一个字符到字典树中
func (t *Trie) Put(word string) {
	curNodes := t.words

	word = strings.ToLower(word)
	for i := 0; i < len(word); i++ {
		ind := word[i] - 'a'
		// 如果当前位置还没有字符
		if curNodes[ind] == nil {
			curNodes[ind] = &node{
				next:  make([]*node, 26),
				isEnd: false,
			}
		}
		if i == len(word)-1 {
			curNodes[ind].isEnd = true
		}
		curNodes = curNodes[ind].next
	}
}

// Find 查找word是否在字典树中
func (t *Trie) Find(word string) bool {
	word = strings.ToLower(word)

	curNodes := t.words

	for i := 0; i < len(word); i++ {
		ind := word[i] - 'a'
		if curNodes[ind] == nil {
			return false
		}
		if i == len(word)-1 {
			return curNodes[ind].isEnd
		}
		curNodes = curNodes[ind].next
	}
	return false
}

func (t *Trie) Prefix(prefix string) bool {
	prefix = strings.ToLower(prefix)

	curNodes := t.words

	for i := 0; i < len(prefix); i++ {
		ind := prefix[i] - 'a'
		if curNodes[ind] == nil {
			return false
		}
		curNodes = curNodes[ind].next
	}
	return true
}
