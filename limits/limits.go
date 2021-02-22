package limits

// IntMax int的最大值
const IntMax = int(^uint(0) >> 1)

// IntMin int的最小值
const IntMin = ^IntMax
