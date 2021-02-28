package algorithm

import (
	"container/heap"
)

type point struct {
	x, y       int
	distance   int
	cost       int
	px, py     int //parent position
	isClosed   bool
	isOpened   bool
	isObstacle bool
}

// ScoreHeap is heap of score
type ScoreHeap []*point

func (h ScoreHeap) Len() int      { return len(h) }
func (h ScoreHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h ScoreHeap) Less(i, j int) bool {
	return (h[i].cost + h[i].distance) < (h[j].cost + h[j].distance)
}

// Push a element
func (h *ScoreHeap) Push(x interface{}) {
	// Push and Pop use pointer receivers because they modify the slice's length,
	// not just its contents.
	*h = append(*h, x.(*point))
}

// Pop and return minimal element
func (h *ScoreHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

// 可选择的下一步方向
var nextX = []int{-1, 0, 1, 0}
var nextY = []int{0, 1, 0, -1}

// TODO：这里可以考虑再定义一个对于每一步操作的代价

// AstarPathWithObstacles find a good but not the shortest path from top left to buttom right
func AstarPathWithObstacles(obstacleGrid [][]int) [][]int {
	var ret [][]int
	m, n := len(obstacleGrid), len(obstacleGrid[0])
	grid := newPointGrid(obstacleGrid)
	open := &ScoreHeap{}

	heap.Push(open, &grid[0][0])

	for open.Len() > 0 {
		s := open.Pop().(*point)
		s.isClosed = true
		if s.isObstacle {
			continue
		}
		if s.x == m-1 && s.y == n-1 {
			// 找到了，然后根据parent属性一路往回找就能找到路径
			return getPath(grid, s)
		}
		for i := 0; i < len(nextX); i++ {
			nx := s.x + nextX[i]
			ny := s.y + nextY[i]

			if nx >= 0 && nx < m && ny >= 0 && ny < n {
				np := &grid[nx][ny]
				if np.isClosed || np.isObstacle {
					continue
				}
				if np.isOpened {
					// 在只能向上下左右四个方向走时，可以认为每一个点的cost+distance都是不变的。
					// 但是这里也写一下对cost的更新
					if np.cost < s.cost+1 {
						np.cost = s.cost + 1
						np.px = s.x
						np.py = s.y
					}
					// 发生了变化，更新heap
					heap.Init(open)
				} else {
					np.isOpened = true
					np.px = s.x
					np.py = s.y
					np.cost = s.cost + 1
					heap.Push(open, np)
				}
			}
		}
	}

	return ret
}
func getPath(grid [][]point, p *point) [][]int {
	path := [][]int{}
	for !(p.x == 0 && p.y == 0) {
		path = append(path, []int{p.x, p.y})
		p = &grid[p.px][p.py]
	}
	path = append(path, []int{0, 0})

	return reverse(path)
}

func reverse(data [][]int) [][]int {
	for i := 0; i < len(data)/2; i++ {
		data[i], data[len(data)-1-i] = data[len(data)-1-i], data[i]
	}
	return data
}
func newPointGrid(obstacleGrid [][]int) [][]point {
	m, n := len(obstacleGrid), len(obstacleGrid[0])
	grid := make([][]point, m)

	for i := 0; i < len(grid); i++ {
		grid[i] = make([]point, n)
	}

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if obstacleGrid[i][j] == 1 {
				grid[i][j].isObstacle = true
			}
			grid[i][j].distance = m + n - i - j
			grid[i][j].x = i
			grid[i][j].y = j
		}
	}
	return grid
}
