package main

import (
	"fmt"
	"strconv"
	"strings"
	"testing"
)

func Test_doRemoveZeroSumSublist(t *testing.T) {
	str := "[1,2,-3,3,1]"

	head, _ := stringToList(str)
	after := removeZeroSumSublists(head)
	fmt.Println(after)
}

func Test_findNthDigit(t *testing.T) {
	var builder strings.Builder
	for i := 0; i < 100; i++ {
		builder.WriteString(strconv.Itoa(i))
	}
	want := builder.String()

	for i := 0; i < len(want); i++ {
		if findNthDigit(i) != int(want[i]-'0') {
			t.Errorf("i = %v, want %v, get %v\n", i, int(want[i]-'0'), findNthDigit(i))
		}
	}
}
