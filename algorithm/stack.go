package algorithm

import "fmt"

// Stack implmentation
type Stack struct {
	data []int
}

//Push a element
func (s *Stack) Push(x int) {
	s.data = append(s.data, x)
}

// Pop element, if Stack is empty, this pop will do nothing
func (s *Stack) Pop() {
	if len(s.data) == 0 {
		return
	}
	s.data = s.data[0 : len(s.data)-1]
}

// Top return the top element in stack, return error if Stack is empty
func (s *Stack) Top() (int, error) {
	if len(s.data) == 0 {
		return 0, fmt.Errorf("attemping to get element from empty stack")
	}

	return s.data[len(s.data)-1], nil
}

// Empty impl
func (s *Stack) Empty() bool {
	return len(s.data) == 0
}
