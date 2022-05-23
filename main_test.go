package main

import (
	"fmt"
	"reflect"
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

func Test_minDeletionSizeOfString(t *testing.T) {
	type args struct {
		str string
	}
	tests := []struct {
		name string
		args args
		want []downRange
	}{
		{
			args: args{
				str: "acbdfea",
			},
			want: []downRange{
				{1, 2},
				{4, 6},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := minDeletionSizeOfString(tt.args.str); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("minDeletionSizeOfString() = %v, want %v", got, tt.want)
			}
		})
	}
}
