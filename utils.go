package main

import (
	"container/list"
	"fmt"
	"strconv"
	"strings"
)

func trimAndSplit(str string, sep string) ([]int, error) {
	var ret []int
	str = strings.TrimFunc(str, func(r rune) bool {
		return r == '[' || r == ']'
	})
	tokens := strings.Split(str, sep)
	ret = make([]int, len(tokens))
	for i := 0; i < len(tokens); i++ {
		val, err := strconv.Atoi(tokens[i])
		if err != nil {
			return nil, err
		}
		ret[i] = val
	}
	return ret, nil
}

// "[e1, e2...]"
func stringToList(str string) (*ListNode, error) {
	str = strings.TrimFunc(str, func(r rune) bool {
		return r == '[' || r == ']'
	})
	ret := &ListNode{}
	p := ret
	tokens := strings.Split(str, ",")
	for _, token := range tokens {
		i, err := strconv.Atoi(token)
		if err != nil {
			return nil, err
		}
		p.Next = &ListNode{
			Val:  i,
			Next: nil,
		}
		p = p.Next
	}
	return ret.Next, nil
}

func listToString(head *ListNode) string {
	var ret strings.Builder
	prefix := '['
	for head != nil {
		fmt.Fprintf(&ret, "%c%d", prefix, head.Val)
		prefix = ','
		head = head.Next
	}
	ret.WriteByte(']')
	return ret.String()
}

func newNode(str string) (*TreeNode, error) {
	if str == "nil" {
		return nil, nil
	}
	val, err := strconv.Atoi(str)
	if err != nil {
		return nil, err
	}
	return &TreeNode{
		Val: val,
	}, nil
}

// str 的格式为"[n1, n2, ...]"
func stringToBinaryTree(str string) (*TreeNode, error) {
	str = strings.TrimFunc(str, func(r rune) bool {
		return r == '[' || r == ']'
	})

	tokens := strings.Split(str, ",")

	if len(tokens) == 0 {
		return nil, nil
	}

	rootVal, err := strconv.Atoi(tokens[0])
	if err != nil {
		return nil, err
	}
	root := &TreeNode{Val: rootVal}

	// 使用队列来做层序遍历构建
	lst := list.New()
	lst.PushBack(root)
	// 遍历到的节点下标
	i := 1
	for lst.Len() != 0 && i < len(tokens) {
		node := lst.Front().Value.(*TreeNode)
		lst.Remove(lst.Front())
		node.Left, err = newNode(tokens[i])
		i++
		if err != nil {
			return nil, err
		}
		if node.Left != nil {
			lst.PushBack(node.Left)
		}
		node.Right, err = newNode(tokens[i])
		i++
		if err != nil {
			return nil, err
		}
		if node.Right != nil {
			lst.PushBack(node.Right)
		}
	}

	return root, nil
}
