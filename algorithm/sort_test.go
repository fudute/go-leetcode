package algorithm

import (
	"math/rand"
	"reflect"
	"sort"
	"testing"
)

var arr, arrSorted []int
var lenght int = 1000

func init() {
	arr = make([]int, lenght)
	arrSorted = make([]int, lenght)
	for i := 0; i < lenght; i++ {
		arr[i] = rand.Int()
	}

	copy(arr, arrSorted)
	sort.Sort(SortByValue(arrSorted))
}
func TestMergeSort(t *testing.T) {
	type args struct {
		A []int
	}
	tests := []struct {
		name string
		args args
		want []int
	}{
		// TODO: Add test cases.
		{name: "MergeSort", args: args{arr}, want: arrSorted},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := MergeSort(tt.args.A); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("MergeSort() = %v, want %v", got, tt.want)
			}
		})
	}
}
