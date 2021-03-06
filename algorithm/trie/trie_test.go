package trie_test

import (
	"testing"

	"github.com/fudute/go-leetcode/algorithm/trie"
	"github.com/fudute/go-leetcode/utils"
)

func TestTrie(t *testing.T) {
	trie := trie.New()

	trie.Put("abc")
	if !trie.Find("abc") {
		t.Errorf("find error")
	}
}

func BenchmarkPut(b *testing.B) {
	trie := trie.New()
	for i := 0; i < b.N; i++ {
		str := utils.RandomString(6, 20)
		trie.Put(str)
		if trie.Find(str) == false {
			b.Errorf("find error")
		}
	}
}
