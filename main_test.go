package main

import (
	"fmt"
	"testing"
)

func Test_doRemoveZeroSumSublist(t *testing.T) {
	str := "[1,2,-3,3,1]"

	head, _ := stringToList(str)
	after := removeZeroSumSublists(head)
	fmt.Println(after)
}
