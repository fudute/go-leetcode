package main

// "Updata-Query Rectangle"的Update操作并没有必要每一次都真正的Update，因为矩阵可能非常大，
// 我们只需要记录Update操作，之后Query的时候，从后向前遍历之前的Update操作，
// 如果命中了，就直接返回那一次操作的newValue。
// 但是这会有个问题，当做了非常多次的Update操作之后，Update日志可能非常长，
// 对于某个长时间没有Update到的值的查询会浪费非常多的时间，
// 可能的优化方式：
// 添加一个数组isUpdated用来表示对应位置的数据是否是最新的，
// 但是每一次更新都会使isUpdated失效，也同样需要更新，并没有优化的效果，
// 所以可以给每个Update操作一个id，然后让isUpdated是否有效和Update的id相关。

type oper struct {
	row1, col1, row2, col2, newValue int
}

// SubrectangleQueries 可以实现快速范围更新和查找范围矩形的实现
type SubrectangleQueries struct {
	rect  [][]int
	opers []oper
}

// New is the constructor of SubrectangleQueries
func New(rectangle [][]int) SubrectangleQueries {
	return SubrectangleQueries{
		rect:  rectangle,
		opers: []oper{},
	}
}

// Update values in subrectangle with new value
func (subRect *SubrectangleQueries) Update(row1 int, col1 int, row2 int, col2 int, newValue int) {

	subRect.opers = append(subRect.opers, oper{
		row1:     row1,
		col1:     col1,
		row2:     row2,
		col2:     col2,
		newValue: newValue,
	})
}

// Query the value at [row, col]
func (subRect *SubrectangleQueries) Query(row int, col int) int {
	for i := len(subRect.opers) - 1; i >= 0; i-- {
		op := subRect.opers[i]
		if op.col1 <= col && op.col2 >= col && op.row1 <= row && op.row2 >= row {
			return op.newValue
		}
	}
	return subRect.rect[row][col]
}
