package main

import (
	"reflect"
	"testing"
)

func Test_stringToList(t *testing.T) {
	str := "[1,2,3]"
	lst, err := stringToList(str)
	if err != nil {
		t.Errorf(err.Error())
	}
	if str == listToString(lst) {
		t.Error("format error")
	}
}

func Test_trimAndSplit(t *testing.T) {
	type args struct {
		str string
		sep string
	}
	tests := []struct {
		name    string
		args    args
		want    []int
		wantErr bool
	}{
		{
			name: "test",
			args: args{
				str: "[1,2,3,4]",
				sep: ",",
			},
			want:    []int{1, 2, 3, 4},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := trimAndSplit(tt.args.str, tt.args.sep)
			if (err != nil) != tt.wantErr {
				t.Errorf("trimAndSplit() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("trimAndSplit() = %v, want %v", got, tt.want)
			}
		})
	}
}
