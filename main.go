package main

import (
	"container/list"
	"fmt"
	"math"
	"math/bits"
	"sort"
	"strconv"
	"strings"
	"unsafe"
)

//ListNode singal linked list
type ListNode struct {
	Val  int
	Next *ListNode
}

// TreeNode definition
type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

const Mod int = 1e9 + 7

func reverseList(head *ListNode) *ListNode {
	var ret *ListNode
	p := head
	for p != nil {
		tmp := p
		p = p.Next
		tmp.Next = ret
		ret = tmp
	}
	return ret
}

func printInerStructOfSlice(s []int) {
	iner := (*[3]int)(unsafe.Pointer(&s))
	fmt.Println("inerstruct of nil slice:", iner)
}

// Len return lenght of list
func getLen(head *ListNode) int {
	len := 0
	for head != nil {
		len++
		head = head.Next
	}
	return len
}
func getIntersectionNode(headA, headB *ListNode) *ListNode {
	pa := headA
	pb := headB

	for pa != pb {
		if pa == nil {
			pa = headB
		} else {
			pa = pa.Next
		}
		if pb == nil {
			pb = headA
		} else {
			pb = pb.Next
		}
	}
	return pa
}

func mergeTwoLists(l1 *ListNode, l2 *ListNode) *ListNode {
	if l1 == nil {
		return l2
	}
	if l2 == nil {
		return l1
	}
	if l1.Val > l2.Val {
		l1, l2 = l2, l1
	}
	ret := l1
	pre := l1
	for l1 != nil && l2 != nil {
		if l1.Val <= l2.Val {
			pre = l1
			l1 = l1.Next
		} else {
			tmp := l2
			l2 = l2.Next
			pre.Next = tmp
			pre = tmp
			tmp.Next = l1
		}
	}
	if l1 == nil {
		pre.Next = l2
	}
	return ret
}

func majorityElement(nums []int) int {
	result, count := nums[0], 1
	for i := 1; i < len(nums); i++ {
		if nums[i] == result {
			count++
		} else {
			count--
			if count == 0 {
				result = nums[i]
				count = 1
			}
		}
	}
	return result
}

func hasCycle(head *ListNode) bool {
	fast, slow := head, head
	for fast != nil && slow != nil {
		fast = fast.Next
		if fast == nil {
			return false
		}
		fast = fast.Next
		slow = slow.Next

		if fast == slow {
			return true
		}
	}
	return false
}

func twoSum(nums []int, target int) []int {
	m := make(map[int]int)
	ret := make([]int, 2)
	for i, key := range nums {
		ind, ok := m[target-key]
		if ok {
			ret[0] = i
			ret[1] = ind
			return ret
		}
		m[key] = i
	}
	return ret
}

// 判断二叉树是否镜像对称
func isSymmetric(root *TreeNode) bool {
	if root == nil {
		return true
	}
	return check(root.Left, root.Right)
}

func check(l *TreeNode, r *TreeNode) bool {
	if l == nil && r == nil {
		return true
	}
	if l == nil || r == nil {
		return false
	}

	return l.Val == r.Val && check(l.Left, r.Right) && check(l.Right, r.Left)
}

func checkPossibility(nums []int) bool {
	left, right := 0, len(nums)-1
	for i := 1; i < len(nums) && nums[i] >= nums[i-1]; i++ {
		left++
	}
	if left == len(nums)-1 {
		return true
	}

	for i := len(nums) - 2; i >= 0 && nums[i] <= nums[i+1]; i-- {
		right--
	}

	if left+1 != right {
		return false
	}
	if left == 0 || right == len(nums)-1 {
		return true
	}
	return nums[left] <= nums[right+1] || nums[left-1] <= nums[right]
}

func sortList(head *ListNode) *ListNode {
	// 归并排序

	// 停止条件
	if head == nil || head.Next == nil {
		return head
	}
	// 1 找到链表的中点 N/2
	mid := findMid(head)
	mid2 := mid.Next
	// 2 分别排序两个子链表
	mid.Next = nil
	l1 := sortList(head)
	l2 := sortList(mid2)
	// 合并排序号的链表
	return mergeTwoLists(l1, l2)
}

// 快慢指针法
func findMid(head *ListNode) *ListNode {
	pre, fast, slow := head, head, head
	for fast != nil {
		fast = fast.Next
		if fast == nil {
			return slow
		}
		fast = fast.Next
		pre = slow
		slow = slow.Next
	}
	return pre
}

func reverse(s []int) []int {
	for i, j := 0, len(s)-1; i < j; i, j = i+1, j-1 {
		s[i], s[j] = s[j], s[i]
	}
	return s
}

// type intPair struct {
// 	val   int // val
// 	index int // index
// }

// // 从后遍历，然后大到下压入栈——单调栈
// func dailyTemperatures(T []int) []int {
// 	ret := make([]int, len(T))
// 	s := make([]intPair, 0)
// 	for i := len(T) - 1; i >= 0; i-- {
// 		tmp := len(s) - 1
// 		for tmp >= 0 && s[tmp].val <= T[i] {
// 			tmp--
// 		}
// 		s = s[0 : tmp+1]
// 		if tmp == -1 {
// 			ret[i] = 0
// 		} else {
// 			ret[i] = s[tmp].index - i
// 		}
// 		s = append(s, intPair{val: T[i], index: i})
// 	}
// 	return ret
// }
func deleteNode(node *ListNode) {
	pre := node
	for node.Next != nil {
		node.Val = node.Next.Val
		pre = node
		node = node.Next
	}
	pre.Next = nil
}

func maxDepth(root *TreeNode) int {
	if root == nil {
		return 0
	}

	left := maxDepth(root.Left)
	right := maxDepth(root.Right)

	var ret int
	if left > right {
		ret = left + 1
	} else {
		ret = right + 1
	}
	return ret
}

func sortedArrayToBST(nums []int) *TreeNode {
	if len(nums) == 0 {
		return nil
	}
	if len(nums) == 1 {
		return &TreeNode{Val: nums[0], Left: nil, Right: nil}
	}
	i := len(nums) / 2
	left := sortedArrayToBST(nums[0:i])
	right := sortedArrayToBST(nums[i+1:])
	return &TreeNode{Val: nums[i], Left: left, Right: right}
}

// func reverseString(s []byte) {
// 	for i := 0; i < len(s)/2; i++ {
// 		s[i], s[len(s)-i-1] = s[len(s)-i-1], s[i]
// 	}
// }

func reverseString(s string) string {
	var builder strings.Builder
	for i := len(s) - 1; i >= 0; i-- {
		builder.WriteByte(s[i])
	}
	return builder.String()
}

func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	result := &ListNode{Val: 0, Next: nil}
	head := result
	c := 0
	for l1 != nil || l2 != nil {
		var left, right int
		if l1 != nil {
			left = l1.Val
			l1 = l1.Next
		}
		if l2 != nil {
			right = l2.Val
			l2 = l2.Next
		}
		sum := left + right + c
		c = sum / 10
		sum = sum % 10

		result.Next = &ListNode{
			Val:  sum,
			Next: nil,
		}
		result = result.Next
	}
	if c != 0 {
		result.Next = &ListNode{
			Val:  c,
			Next: nil,
		}
	}
	return head.Next
}

func lengthOfLongestSubstring(s string) int {
	left, right := 0, 0 // left表示窗口的左边界，right表示窗口的右边界（包含）
	ret := 0
	leastPos := make([]int, 128) // 记录26个字母上一次出现的位置
	count := make([]int, 128)    // 记录当前窗口中字母的出现个数
	for right < len(s) {
		if count[s[right]] == 0 { //上一个窗口中没有当前字母
			count[s[right]] = 1
			leastPos[s[right]] = right
			if ret < right-left+1 {
				ret = right - left + 1
			}
			right++
		} else {
			// 之前的窗口中存在当前字母，那么直接将left调节到leastPos[s[right]] + 1
			for i := left; i < leastPos[s[right]]; i++ {
				count[s[i]] = 0
			}
			left = leastPos[s[right]] + 1
			leastPos[s[right]] = right
			right++
		}
	}
	return ret
}

func min(nums ...int) int {
	ret := math.MaxInt64
	for i := 0; i < len(nums); i++ {
		if nums[i] < ret {
			ret = nums[i]
		}
	}
	return ret
}
func maxArea(height []int) int {
	left, right := 0, len(height)-1
	ret := 0

	for left < right {

		area := min(height[left], height[right]) * (right - left)
		if area > ret {
			ret = area
		}
		if height[left] > height[right] {
			right--
		} else {
			left++
		}
	}
	return ret
}

func checkInclusion(s1 string, s2 string) bool {
	len1, len2 := len(s1), len(s2)
	if len1 > len2 {
		return false
	}
	templ := make([]int, 26)
	for _, c := range s1 {
		templ[c-'a']++
	}
	count := make([]int, 27)
	for i := 0; i < len1; i++ {
		count[s2[i]-'a']++
	}

	isSame := func(lhs, rhs []int) bool {
		for i := 0; i < len(lhs); i++ {
			if lhs[i] != rhs[i] {
				return false
			}
		}
		return true
	}
	if isSame(templ, count) {
		return true
	}

	for i := 0; i < len2-len1; i++ {
		count[s2[i]-'a']--
		count[s2[i+len1]-'a']++
		if isSame(templ, count) {
			return true
		}
	}
	return false
}

func reachNumber(target int) int {
	if target == 0 {
		return 0
	}
	m := make(map[int]struct{})
	m[0] = struct{}{}
	iteration := 1
	for {
		tmp := make(map[int]struct{})
		for k := range m {
			// 向左和向右走
			if k+iteration == target {
				return iteration
			}
			if k-iteration == target {
				return iteration
			}
			tmp[k+iteration] = struct{}{}
			tmp[k-iteration] = struct{}{}
		}
		iteration++
		m = tmp
	}
}

func hammingWeight(num uint32) int {
	return bits.OnesCount32(num)
}

func singleNumber(nums []int) int {
	var ret int
	for _, v := range nums {
		ret = ret ^ v
	}
	return ret
}

func permute(nums []int) [][]int {
	if len(nums) == 1 {
		return [][]int{nums}
	}
	res := [][]int{}
	visited := make([]int, len(nums))

	// 这里我定义了一个内部函数变量，这样做的好处是可以共享permute函数内的变量，减少参数的传递
	var backtrack func(int, []int)
	backtrack = func(pos int, output []int) {
		if pos == len(nums) {
			tmp := make([]int, len(nums))
			copy(tmp, output)
			res = append(res, tmp)
			return
		}

		for i := 0; i < len(nums); i++ {
			if visited[i] == 0 {
				visited[i] = 1
				backtrack(pos+1, append(output, nums[i]))
				visited[i] = 0
			}
		}
	}

	output := make([]int, 0)
	backtrack(0, output)

	return res
}

func permuteUnique(nums []int) [][]int {
	if len(nums) == 1 {
		return [][]int{nums}
	}
	res := [][]int{}
	visited := make([]int, len(nums))

	var backtrack func(int, []int)
	backtrack = func(pos int, output []int) {
		if pos == len(nums) {
			tmp := make([]int, len(nums))
			copy(tmp, output)
			res = append(res, tmp)
			return
		}

		// 在这里，增加一个防止重复的set,记录在当前ind已经安排了哪些值
		m := make(map[int]struct{})
		for i := 0; i < len(nums); i++ {
			if visited[i] == 0 {
				if _, ok := m[nums[i]]; ok {
					continue
				}
				visited[i] = 1
				backtrack(pos+1, append(output, nums[i]))
				visited[i] = 0
				m[nums[i]] = struct{}{}
			}
		}
	}

	backtrack(0, []int{})

	// sort.Slice(res, func(i, j int) bool {
	// 	for k := 0; k < len(res[0]); k++ {
	// 		if res[i][k] != res[j][k] {
	// 			return res[i][k] < res[j][k]
	// 		}
	// 	}
	// 	return len(res[i]) < len(res[j])
	// })
	return res
}
func reverseWords(s string) string {
	var rev []string
	words := strings.Split(s, " ")
	for _, w := range words {
		rev = append(rev, reverseString(w))
	}
	return strings.Join(rev, " ")
}

func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}
	// p 是更小的那个
	if p.Val > q.Val {
		p, q = q, p
	}
	// 只要找到节点t，满足p.Val <= t.Val <= q.Val
	// 可以使用链表 list 来实现队列或者栈
	ls := list.New()
	ls.PushBack(root)
	for ls.Len() > 0 {
		node := ls.Remove(ls.Front()).(*TreeNode)
		if p.Val <= node.Val && node.Val <= q.Val {
			return node
		}
		if node.Left != nil {
			ls.PushBack(node.Left)
		}
		if node.Right != nil {
			ls.PushBack(node.Right)
		}
	}
	return nil
}

func preOrder(root *TreeNode) {
	ls := list.New()
	ls.PushBack(root)
	for ls.Len() > 0 {
		node := ls.Remove(ls.Front()).(*TreeNode)
		fmt.Printf("%d ", node.Val)
		if node.Left != nil {
			ls.PushBack(node.Left)
		}
		if node.Right != nil {
			ls.PushBack(node.Right)
		}
	}
}
func productExceptSelf(nums []int) []int {
	ret := make([]int, len(nums))
	ret[0] = 1

	for i := 1; i < len(nums); i++ {
		ret[i] = ret[i-1] * nums[i-1]
	}

	post := 1
	for i := len(nums) - 2; i >= 0; i-- {
		post = post * nums[i+1]
		ret[i] *= post
	}

	return ret
}

func isValid(s string) bool {
	stack := make([]byte, 0)
	for i := 0; i < len(s); i++ {
		if len(stack) == 0 {
			stack = append(stack, s[i])
		} else if s[i] == '(' || s[i] == '{' || s[i] == '[' {
			stack = append(stack, s[i])
		} else {
			if s[i] == ')' && stack[len(stack)-1] == '(' ||
				s[i] == '}' && stack[len(stack)-1] == '{' ||
				s[i] == ']' && stack[len(stack)-1] == '[' {

				stack = stack[0 : len(stack)-1]
			} else {
				return false
			}
		}
	}
	return len(stack) == 0
}

func containsDuplicate(nums []int) bool {
	m := make(map[int]struct{})
	for _, key := range nums {
		_, ok := m[key]
		if ok {
			return true
		}
		m[key] = struct{}{}
	}
	return false
}
func rotateRight(head *ListNode, k int) *ListNode {
	if head == nil {
		return nil
	}
	if k == 0 {
		return head
	}
	lenght := 0
	p := head
	for p = head; p != nil; p = p.Next {
		lenght++
	}

	k = k % lenght
	if k == 0 {
		return head
	}

	p = head
	for i := 0; i < lenght-k-1; i++ {
		p = p.Next
	}

	newHead := p.Next
	p.Next = nil

	q := newHead
	for q.Next != nil {
		q = q.Next
	}
	q.Next = head
	return newHead
}

func multiply(num1 string, num2 string) string {
	ret := make([]byte, 0)
	// 使num1是长度较小的那个
	if len(num1) > len(num2) {
		num1, num2 = num2, num1
	}
	b2 := []byte(num2)
	for i := 0; i < len(num1); i++ {
		ret = append(ret, '0')
		tmp := multiplyOneNum(b2, int(num1[i]-'0'))
		ret = add(ret, tmp)
	}

	for len(ret) > 1 && ret[0] == '0' {
		ret = ret[1:]
	}
	return string(ret)
}

func multiplyOneNum(num []byte, a int) []byte {
	c := 0
	ret := make([]byte, len(num)+1)
	for i := 0; i < len(num); i++ {
		mult := int(num[len(num)-1-i]-'0')*a + c
		c = mult / 10
		mult = mult % 10
		ret[len(ret)-i-1] = byte(mult) + '0'
	}
	if c == 0 {
		return ret[1:]
	}
	ret[0] = byte(c) + '0'
	return ret
}

// 两个非负整数求和，num1和num2中不存在前导0
func add(num1, num2 []byte) []byte {
	c := 0
	len1, len2 := len(num1), len(num2)
	if len1 < len2 {
		len1, len2 = len2, len1
		num1, num2 = num2, num1
	}

	ret := make([]byte, len1+1)
	for i := 0; i < len1; i++ {
		l := int(num1[len1-i-1] - '0')
		var r int
		if i < len2 {
			r = int(num2[len2-i-1] - '0')
		}
		sum := r + l + c
		c = sum / 10
		sum = sum % 10
		ret[len1-i] = byte(sum) + '0'
	}
	if c == 0 {
		return ret[1:]
	}
	ret[0] = byte(c) + '0'
	return ret
}

func uniquePaths(m int, n int) int {
	grid := make([][]int, n)
	for i := 0; i < n; i++ {
		grid[i] = make([]int, m)
	}

	// 1 初始化表格
	for i := 0; i < m; i++ {
		grid[n-1][i] = 1
	}

	for i := 0; i < n; i++ {
		grid[i][m-1] = 1
	}

	// 2 从右下到左上，依次赋值
	for i := n - 2; i >= 0; i-- {
		for j := m - 2; j >= 0; j-- {
			grid[i][j] = grid[i+1][j] + grid[i][j+1]
		}
	}
	return grid[0][0]
}

func findDisappearedNumbers(nums []int) []int {
	// 我想到了将数组中元素的值作为需要操作的元素的下标，但是我想的是赋值，这会导致丢失被赋值的位置原本的信息
	// 然后正确的做法是将被操作元素取负值，这样就不会丢失信息
	ret := make([]int, 0)
	for i := 0; i < len(nums); i++ {
		var ind int
		if nums[i] > 0 {
			ind = nums[i] - 1
		} else {
			ind = -nums[i] - 1
		}

		if nums[ind] > 0 {
			nums[ind] = -nums[ind]
		}
	}
	for i := 0; i < len(nums); i++ {
		if nums[i] > 0 {
			ret = append(ret, i+1)
		}
	}
	return ret
}

func minSwapsCouples(row []int) int {
	return 0
}

func matrixReshape(nums [][]int, r int, c int) [][]int {
	if nums == nil {
		return nil
	}

	m, n := len(nums), len(nums[0])

	if m*n != r*c {
		return nums
	}

	ret := make([][]int, r)
	for i := 0; i < r; i++ {
		ret[i] = make([]int, c)
	}

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			a, b := (i*n+j)/c, (i*n+j)%c
			ret[a][b] = nums[i][j]
		}
	}
	return ret
}

// 查找两个有序链表的中位数
func findMedianSortedArrays(nums1 []int, nums2 []int) float64 {

	len1, len2 := len(nums1), len(nums2)

	half := (len1 + len2 + 1) / 2

	if len1 > len2 {
		len1, len2 = len2, len1
		nums1, nums2 = nums2, nums1
	}

	// 二分查找寻找分割点
	low, high := 0, len1
	// 表示在下标为mid1之前的位置画一条分割线，前半部分有mid1个元素
	mid1 := (low + high) / 2
	mid2 := half - mid1
	for low <= high {
		l1, l2, r1, r2 := math.MinInt32, math.MinInt32, math.MaxInt32, math.MaxInt32

		if mid1-1 >= 0 {
			l1 = nums1[mid1-1]
		}
		if mid1 < len1 {
			r1 = nums1[mid1]
		}
		if mid2-1 >= 0 {
			l2 = nums2[mid2-1]
		}
		if mid2 < len2 {
			r2 = nums2[mid2]
		}
		if l1 <= r2 && r1 >= l2 {
			// 找到分割点
			if (len1+len2)%2 == 1 {
				if l1 > l2 {
					return float64(l1)
				}
				return float64(l2)
			}
			l, r := l1, r1
			if l < l2 {
				l = l2
			}
			if r > r2 {
				r = r2
			}
			return float64(r+l) / 2

		}

		if l1 > r2 {
			high = mid1 - 1
		} else {
			low = mid1 + 1
		}

		mid1 = (low + high) / 2
		mid2 = half - mid1
	}
	return 0
}

func findMaxConsecutiveOnes(nums []int) int {
	var ret, count int

	for i := 0; i < len(nums); i++ {
		if nums[i] == 1 {
			count++
		} else {
			if count > ret {
				ret = count
			}
			count = 0
		}
	}
	return ret
}

// 最大连续1的个数Ⅲ
// 只要区间[left, right]内0的个数小于K，那么就可以构造出长度为right-left的区间
// 所以就采用双指针的方式
func longestOnes(A []int, K int) int {
	var left, right, count0, ret int
	for right < len(A) {
		if A[right] == 0 {
			if count0 < K {
				count0++
			} else {
				for A[left] == 1 {
					left++
				}
				// 去掉一个0
				left++
			}
		}
		right++
		if right-left > ret {
			ret = right - left
		}
	}
	return ret
}

func runningSum(nums []int) []int {
	if nums == nil || len(nums) == 0 {
		return nums
	}
	ret := make([]int, len(nums))
	ret[0] = nums[0]
	for i := 1; i < len(nums); i++ {
		ret[i] = ret[i-1] + nums[i]
	}
	return ret
}

func removeZeroSumSublists(head *ListNode) *ListNode {
	ret, ok := doRemoveZeroSumSublist(head)
	for ok {
		ret, ok = doRemoveZeroSumSublist(ret)
	}
	return ret
}
func doRemoveZeroSumSublist(head *ListNode) (*ListNode, bool) {
	// 需要在链表的前面添加一个val为0的头结点，方便后面返回
	ret := &ListNode{
		Val:  0,
		Next: head,
	}
	p := ret

	// sum -- Listnode
	m := make(map[int]*ListNode)

	sum := 0
	for p != nil {
		sum += p.Val

		node, ok := m[sum]
		if ok {
			// 说明从node到当前节点之间的所有节点和为0，删除
			node.Next = p.Next
			return ret.Next, true
		}
		m[sum] = p
		p = p.Next
	}
	return ret.Next, false
}

func isToeplitzMatrix(matrix [][]int) bool {
	m, n := len(matrix), len(matrix[0])

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if i-1 >= 0 && j-1 >= 0 && matrix[i-1][j-1] != matrix[i][j] {
				return false
			}
		}
	}
	return true
}

func binarySearch(nums []int, target int) int {
	low, high := 0, len(nums)-1
	for low <= high {
		mid := (low + high) / 2
		if nums[mid] == target {
			return mid
		}
		if nums[mid] < target {
			low = mid + 1
		}
		high = mid - 1
	}
	return -1
}

// 对于nums中的每一个元素，调用condition(i)之后得到的结果集合应该满足这样的形式:
// [1, 1, 1, 0, -1, -1, -1]
func conditionalBinarySearch(nums []int, condition func(ind int) int) int {
	low, high := 0, len(nums)-1
	for low <= high {
		mid := (low + high) / 2
		switch condition(mid) {
		case 0:
			return mid
		case 1:
			low = mid + 1
		case -1:
			high = mid - 1
		}
	}
	return -1
}
func searchRange(nums []int, target int) []int {
	ret := make([]int, 2)
	ret[0] = conditionalBinarySearch(nums, func(ind int) int {
		if nums[ind] == target && (ind == 0 || nums[ind-1] < target) {
			return 0
		}
		if nums[ind] < target {
			return 1
		}
		return -1
	})
	ret[1] = conditionalBinarySearch(nums, func(ind int) int {
		if nums[ind] == target && (ind == len(nums)-1 || nums[ind+1] > target) {
			return 0
		}
		if nums[ind] <= target {
			return 1
		}
		return -1
	})
	return ret
}

func searchRange2(nums []int, target int) []int {
	for i := range nums {
		if nums[i] == target {
			j := i + 1
			for j < len(nums) {
				if nums[j] == target {
					j++
				} else {
					break
				}
			}
			return []int{i, j - 1}
		}
	}
	return []int{-1, -1}
}

func fib(n int) int {
	if n < 2 {
		return n
	}

	fibs := make([]int, n+1)
	fibs[0] = 0
	fibs[1] = 1

	for i := 2; i <= n; i++ {
		fibs[i] = fibs[i-1] + fibs[i-2]
	}
	return fibs[n]
}

var longestPath int

func longestUnivaluePath(root *TreeNode) int {
	longestPath = 0
	longestUnivaluePathWithRoot(root)
	return longestPath
}

func longestUnivaluePathWithRoot(root *TreeNode) int {
	if root == nil {
		return 0
	}
	// left 从当前根节点向左子树延伸的单值最短路径
	var left, right int
	if root.Left != nil {
		if root.Left.Val == root.Val {
			left = longestUnivaluePathWithRoot(root.Left) + 1
		} else {
			longestUnivaluePathWithRoot(root.Left)
		}
	}

	if root.Right != nil {
		if root.Right.Val == root.Val {
			right = longestUnivaluePathWithRoot(root.Right) + 1
		} else {
			longestUnivaluePathWithRoot(root.Right)
		}
	}
	if left+right > longestPath {
		longestPath = left + right
	}

	if left > right {
		return left
	}
	return right
}

type nodeInfo struct {
	col   int
	row   int
	value int
}

type verticalTraversalSort []nodeInfo

func (a verticalTraversalSort) Len() int      { return len(a) }
func (a verticalTraversalSort) Swap(i, j int) { a[i], a[j] = a[j], a[i] }
func (a verticalTraversalSort) Less(i, j int) bool {
	if a[i].col != a[j].col {
		return a[i].col < a[j].col
	}
	if a[i].row != a[j].row {
		return a[i].row < a[j].row
	}
	return a[i].value < a[j].value
}

var nodes []nodeInfo

func verticalTraversal(root *TreeNode) [][]int {
	nodes = make([]nodeInfo, 0)
	inOrder(root, 0, 0)
	sort.Sort(verticalTraversalSort(nodes))

	ret := make([][]int, 0)
	i := 0
	for i < len(nodes) {
		ret = append(ret, make([]int, 0))
		last := len(ret) - 1
		ret[last] = append(ret[last], nodes[i].value)
		i++
		for i < len(nodes) && nodes[i].col == nodes[i-1].col {
			ret[last] = append(ret[last], nodes[i].value)
			i++
		}
	}
	return ret
}

func inOrder(root *TreeNode, row, col int) {
	if root == nil {
		return
	}
	nodes = append(nodes, nodeInfo{
		col:   col,
		row:   row,
		value: root.Val,
	})

	inOrder(root.Left, row+1, col-1)
	inOrder(root.Right, row+1, col+1)
}

func maxSatisfied(customers []int, grumpy []int, X int) int {
	ret, satisfiied := 0, 0
	for i := 0; i < len(customers); i++ {
		if grumpy[i] == 0 {
			satisfiied += customers[i]
		}
	}

	for i := 0; i < X; i++ {
		if grumpy[i] == 1 {
			satisfiied += customers[i]
		}
	}
	ret = satisfiied
	for i := X; i < len(customers); i++ {
		if grumpy[i] == 1 {
			satisfiied += customers[i]
		}
		if grumpy[i-X] == 1 {
			satisfiied -= customers[i-X]
		}
		if satisfiied > ret {
			ret = satisfiied
		}
	}
	return ret
}

func exchangeBits(num int) int {
	// 奇数位右移，偶数位左移
	odd := num >> 1 & 0x55555555
	even := num << 1 & 0xaaaaaaaa
	return odd | even
}

func subarraySum(nums []int, k int) int {
	m := make(map[int]int)
	m[0] = 1
	subSum := 0
	ret := 0
	for i := 0; i < len(nums); i++ {
		subSum += nums[i]
		times, ok := m[subSum-k]
		if ok {
			ret += times
		}
		m[subSum]++
	}
	return ret
}
func canJump(nums []int) bool {
	cur := 0
	for {
		// 判断能否跳到终点
		if nums[cur]+cur >= len(nums)-1 {
			return true
		}
		// 如果当前位置能跳的步数为0，则无法到达终点
		if nums[cur] == 0 {
			return false
		}
		// next表示接下来该跳多少步
		next, maxStep := 0, 0
		// i表示跳多少步
		for i := 1; i <= nums[cur]; i++ {
			if maxStep <= nums[cur+i]+i {
				maxStep = nums[cur+i] + i
				next = i
			}
		}
		cur += next
	}
}

func findNthDigit(n int) int {
	if n < 10 {
		return n
	}
	n -= 10

	i := 2
	for ; n >= 9*i*int(math.Pow10(i-1)); i++ {
		n = n - 9*i*int(math.Pow10(i-1))
	}

	num := n/i + int(math.Pow10(i-1))
	index := n % i
	numStr := strconv.Itoa(num)
	return int(numStr[index] - '0')
}

func flipAndInvertImage(A [][]int) [][]int {
	for i := 0; i < len(A); i++ {
		for j := 0; j < len(A[i])/2; j++ {
			A[i][j], A[i][len(A[i])-j-1] = (A[i][len(A[i])-j-1]+1)%2, (A[i][j]+1)%2
		}
	}

	if len(A[0])%2 == 1 {
		for i := 0; i < len(A); i++ {
			A[i][len(A[i])/2] = (A[i][len(A[i])/2] + 1) % 2
		}
	}
	return A
}

func mergeAlternately(word1 string, word2 string) string {
	var builder strings.Builder
	var isChanged = false
	if len(word1) > len(word2) {
		word1, word2 = word2, word1
		isChanged = true
	}

	for i := 0; i < len(word1); i++ {
		if !isChanged {
			builder.WriteByte(word1[i])
			builder.WriteByte(word2[i])
		} else {
			builder.WriteByte(word2[i])
			builder.WriteByte(word1[i])
		}
	}

	for i := len(word1); i < len(word2); i++ {
		builder.WriteByte(word2[i])
	}

	return builder.String()
}

func minOperations(boxes string) []int {
	left, right := 0, 0

	weight := 0
	for i := 0; i < len(boxes); i++ {
		if boxes[i] == '1' {
			right++
			weight += i
		}
	}

	ret := make([]int, len(boxes))
	ret[0] = weight
	for i := 1; i < len(boxes); i++ {
		if boxes[i-1] == '1' {
			left++
			right--
		}
		weight = weight - right + left
		ret[i] = weight
	}
	return ret
}

func maximumScore(nums []int, multipliers []int) int {
	ret := 0
	if len(multipliers) == 0 {
		return ret
	}
	left := multipliers[0]*nums[0] + maximumScore(nums[1:], multipliers[1:])
	right := multipliers[0]*nums[len(nums)-1] + maximumScore(nums[0:len(nums)-1], multipliers[1:])

	if left > right {
		return left
	}
	return right
}

func combinationSum(candidates []int, target int) [][]int {
	if target == 0 {
		// 存一个空元素，说明能够找到
		return [][]int{
			{},
		}
	}
	if len(candidates) == 0 || target < 0 {
		return [][]int{}
	}

	// 不选择当前元素
	A := combinationSum(candidates[1:], target)
	// 选择当前元素
	B := combinationSum(candidates, target-candidates[0])

	for i := 0; i < len(B); i++ {
		B[i] = append(B[i], candidates[0])
	}
	ret := append(A, B...)
	return ret
}

func combinationSum2(candidates []int, target int) [][]int {
	ret := [][]int{}

	sort.Ints(candidates)

	var backtrack func(i, target int, output []int)

	backtrack = func(i, target int, output []int) {
		if target < 0 {
			return
		}
		if target == 0 {
			cp := make([]int, len(output))
			copy(cp, output)
			ret = append(ret, cp)
			return
		}

		if i < len(candidates) {
			j := i + 1

			for j < len(candidates) && candidates[j] == candidates[j-1] {
				j++
			}

			backtrack(i+1, target-candidates[i], append(output, candidates[i]))
			backtrack(j, target, output)
		}
	}

	backtrack(0, target, []int{})

	return ret
}
func transpose(matrix [][]int) [][]int {
	if len(matrix) == 0 {
		return matrix
	}
	m, n := len(matrix), len(matrix[0])

	ret := make([][]int, n)
	for i := 0; i < n; i++ {
		ret[i] = make([]int, m)
	}

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			ret[j][i] = matrix[i][j]
		}
	}
	return ret
}

func generateParenthesis(n int) []string {

	res := []string{}

	var putParenthesis func(int, int, []byte)
	putParenthesis = func(left, right int, output []byte) {
		if right == 0 {
			res = append(res, string(output))
		}

		if left > 0 {
			putParenthesis(left-1, right, append(output, '('))
		}
		if right > left {
			putParenthesis(left, right-1, append(output, ')'))
		}
	}

	putParenthesis(n, n, []byte{})
	return res
}

func findNumOfValidWords(words []string, puzzles []string) []int {
	const puzzleLength = 7
	cnt := map[int]int{}
	for _, s := range words {
		mask := 0
		for _, ch := range s {
			mask |= 1 << (ch - 'a')
		}
		if bits.OnesCount(uint(mask)) <= puzzleLength {
			cnt[mask]++
		}
	}

	ans := make([]int, len(puzzles))
	for i, s := range puzzles {
		first := 1 << (s[0] - 'a')

		// 枚举子集方法一
		//for choose := 0; choose < 1<<(puzzleLength-1); choose++ {
		//    mask := 0
		//    for j := 0; j < puzzleLength-1; j++ {
		//        if choose>>j&1 == 1 {
		//            mask |= 1 << (s[j+1] - 'a')
		//        }
		//    }
		//    ans[i] += cnt[mask|first]
		//}

		// 枚举子集方法二
		mask := 0
		for _, ch := range s[1:] {
			mask |= 1 << (ch - 'a')
		}
		subset := mask
		for {
			ans[i] += cnt[subset|first]
			subset = (subset - 1) & mask
			if subset == mask {
				break
			}
		}
	}
	return ans
}

func subsets(nums []int) [][]int {
	ret := [][]int{}

	var helper func(pos int, output []int)

	helper = func(pos int, output []int) {
		if pos == len(nums) {
			tmp := make([]int, len(output))
			copy(tmp, output)
			ret = append(ret, tmp)
			return
		}
		helper(pos+1, append(output, nums[pos]))
		helper(pos+1, output)
	}

	helper(0, []int{})
	return ret
}

func checkSubTree(t1 *TreeNode, t2 *TreeNode) bool {
	var self, left, right bool

	self = isSameTree(t1, t2)
	if t1 != nil {
		left = checkSubTree(t1.Left, t2)
		right = checkSubTree(t1.Right, t2)
	}
	return self || left || right
}

func isSameTree(t1, t2 *TreeNode) bool {
	if t1 == nil && t2 == nil {
		return true
	}
	if t1 == nil || t2 == nil {
		return false
	}
	if t1.Val == t2.Val {
		return isSameTree(t1.Left, t2.Left) && isSameTree(t1.Right, t2.Right)
	}
	return false
}
func waysToStep(n int) int {
	// dp[i]表示有i个台阶的上法
	dp := make([]int, 4)
	dp[1] = 1
	dp[2] = 2
	dp[3] = 4

	for i := 4; i < n+1; i++ {
		dp = append(dp, (dp[i-1]+dp[i-2]+dp[i-3])%1000000007)
	}
	return dp[n]
}

func pathWithObstacles(obstacleGrid [][]int) [][]int {
	var ret [][]int
	m, n := len(obstacleGrid), len(obstacleGrid[0])

	visted := make([][]bool, len(obstacleGrid))
	for i := 0; i < len(visted); i++ {
		visted[i] = make([]bool, len(obstacleGrid[0]))
	}

	var move func(x, y int, path [][]int) bool
	move = func(x, y int, path [][]int) bool {
		if visted[x][y] {
			return false
		}
		if obstacleGrid[x][y] == 1 {
			return false
		}
		path = append(path, []int{x, y})
		if x == m-1 && y == n-1 {
			ret = path
			return true
		}
		if x+1 < m {
			if move(x+1, y, path) {
				return true
			}
		}
		if y+1 < n {
			if move(x, y+1, path) {
				return true
			}
		}

		visted[x][y] = true
		return false
	}
	move(0, 0, [][]int{})
	return ret
}

func reverseArray(data [][]int) [][]int {
	for i := 0; i < len(data)/2; i++ {
		data[i], data[len(data)-1-i] = data[len(data)-1-i], data[i]
	}
	return data
}

/*
解题思路：
遍历一遍数组，如果所有的字符都个数大于等于k，结果就是len(s)，
如果某个字母s[i]不满足条件，那么只要在i左右两边执行上面相同的操作
*/
func longestSubstring(s string, k int) int {
	// count[i]包含前i个字母的分布, 前i=0个字母的count[0]为全0
	count := make([][]int, len(s)+1)
	for i := 0; i < len(count); i++ {
		count[i] = make([]int, 26)
	}
	subCount := make([]int, 26)
	for i := 0; i < len(s); i++ {
		ind := s[i] - 'a'
		subCount[ind]++
		copy(count[i+1], subCount)
	}

	var getLongestSubString func(left, right int) int

	getLongestSubString = func(left, right int) int {
		ret := 0
		if left == right {
			return 0
		}
		// 获得[left, right)区间内的字母出现次数
		subCount := make([]int, 26)
		for i := 0; i < 26; i++ {
			subCount[i] = count[right][i] - count[left][i]
		}
		pre := left
		for i := 0; i < 26; i++ {
			if subCount[i] < k {
				// 找到字母 subCount[i]+'a'在区间[pre, right)内的第一次出现位置first
				// 然后只需要考虑[pre, first)区间是否满足条件
				// 只尝试大于当前最大值的范围

				if pre-i > ret {
					tmp := getLongestSubString(pre, i)
					if ret < tmp {
						ret = tmp
					}
				}
				pre = i + 1
			}
		}
		return 0
	}

	return getLongestSubString(0, len(s))
}

func climbStairs(n int) int {
	if n <= 2 {
		return n
	}
	dp := make([]int, n+1)

	dp[1] = 1
	dp[2] = 2

	for i := 3; i < len(dp); i++ {
		dp[i] = dp[i-1] + dp[i-2]
	}
	return dp[n]
}

func numTilePossibilities(tiles string) int {
	ret := 0

	visited := make([]bool, len(tiles))

	var backtrack func(pos int)
	backtrack = func(pos int) {
		ret++
		if pos == len(tiles) {
			return
		}

		// 表示当前位置已经选择过得字符
		choosedInCurrentPos := make([]bool, 26)
		for i := 0; i < len(tiles); i++ {
			ind := tiles[i] - 'A'
			if !visited[i] && !choosedInCurrentPos[ind] {
				visited[i] = true
				backtrack(pos + 1)
				visited[i] = false
				choosedInCurrentPos[ind] = true
			}
		}
	}

	backtrack(0)
	// 排除空选择
	return ret - 1
}

func minFlips(a int, b int, c int) int {
	ret := 0
	for i := 0; i < 32; i++ {
		ai := a >> i & 1
		bi := b >> i & 1
		ci := c >> i & 1

		if ci == 0 {
			ret += ai + bi
		} else if ai+bi == 0 {
			ret++
		}
	}
	return ret
}

func findErrorNums(nums []int) []int {
	var dup, miss int
	for i := 0; i < len(nums); i++ {
		var ind int
		if nums[i] > 0 {
			ind = nums[i] - 1
		} else {
			ind = -nums[i] - 1
		}

		if nums[ind] < 0 {
			dup = ind + 1
		} else {
			nums[ind] *= -1
		}
	}

	for i := 0; i < len(nums); i++ {
		if nums[i] > 0 {
			miss = i + 1
		}
	}
	return []int{dup, miss}
}

func largestDivisibleSubset(nums []int) []int {
	if nums == nil || len(nums) == 0 {
		return []int{}
	}
	// 首先sort一下，简化后面的讨论
	sort.Ints(nums)

	// dp[i] 表示已i结尾的最大整除子集
	dp := make([]int, len(nums))
	dp[0] = 1

	// 长度最长的下标
	last := 0
	for i := 1; i < len(nums); i++ {
		dp[i] = 1
		for j := i - 1; j >= 0; j-- {
			if nums[i]%nums[j] == 0 && dp[j]+1 > dp[i] {
				dp[i] = dp[j] + 1
				if dp[i] > dp[last] {
					last = i
				}
			}
		}
	}

	ret := make([]int, dp[last])

	ret[len(ret)-1] = nums[last]
	for i := len(ret) - 2; i >= 0; i-- {
		for j := last; j >= 0; j-- {
			if dp[last] == dp[j]+1 && nums[last]%nums[j] == 0 {
				ret[i] = nums[j]
				last = j
				break
			}
		}
	}
	return ret
}

func isMonotonic(A []int) bool {
	if A[0] < A[len(A)-1] {
		for i := 1; i < len(A); i++ {
			if A[i]-A[i-1] < 0 {
				return false
			}
		}
	} else {
		for i := len(A) - 2; i >= 0; i-- {
			if A[i]-A[i+1] < 0 {
				return false
			}
		}
	}
	return true
}

func searchBST(root *TreeNode, val int) *TreeNode {
	if root == nil {
		return nil
	}

	if root.Val == val {
		return root
	}

	if root.Val > val {
		return searchBST(root.Left, val)
	}
	return searchBST(root.Right, val)

}

func threeSum(nums []int) [][]int {
	ret := [][]int{}
	sort.Ints(nums)
	for first := 0; first < len(nums); first++ {
		// 每一次都取不同的数
		if first == 0 || nums[first] != nums[first-1] {
			third := len(nums) - 1
			for second := first + 1; second < len(nums); second++ {
				if second == first+1 || nums[second] != nums[second-1] {
					for third > second && nums[first]+nums[second]+nums[third] > 0 {
						third--
					}
					if third > second && nums[first]+nums[second]+nums[third] == 0 {
						ret = append(ret, []int{nums[first], nums[second], nums[third]})
					}
				}
			}
		}
	}
	return ret
}

func duplicateZeros(arr []int) {
	res := make([]int, len(arr))
	ind := 0

	for i := 0; i < len(arr) && ind < len(res); i++ {
		if arr[i] == 0 {
			res[ind] = 0
			if ind+1 < len(res) {
				res[ind+1] = 0
			}
			ind += 2
		} else {
			res[ind] = arr[i]
			ind++
		}
	}
	copy(arr, res)
}

func rotateString(A string, B string) bool {
	if len(A) != len(B) {
		return false
	}

	return strings.Contains(A+B, B)
}

func countBits(num int) []int {
	if num == 0 {
		return []int{0}
	}
	ret := make([]int, num+1)
	ret[1] = 1
	for i := 2; i <= num; i++ {
		if i%2 == 0 {
			ret[i] = ret[i/2]
		} else {
			ret[i] = ret[i-1] + 1
		}
	}
	// for i := 0; i <= num; i++ {
	// 	ret[i] = bits.OnesCount(uint(i))
	// }
	return ret
}

func reverseInt(x int) int {
	var sb strings.Builder
	fmt.Fprintln(&sb, x)

	bs := []byte(sb.String())
	for i := 0; i < len(bs)/2; i++ {
		bs[i], bs[len(bs)-1-i] = bs[len(bs)-1-i], bs[i]
	}
	ret := 0

	for i := 0; i < len(bs); i++ {
		ret = ret*10 + int(bs[i]-'0')
		if ret < 0 {
			return 0
		}
	}
	return ret
}

// 最长递增子序列
func lengthOfLIS(nums []int) int {
	dp := make([]int, len(nums))
	dp[0] = 1
	for i := 1; i < len(dp); i++ {
		dp[i] = 1
		for j := i - 1; j >= 0; j-- {
			if nums[i] > nums[j] && dp[i] < dp[j]+1 {
				dp[i] = dp[j] + 1
			}
		}
	}

	max := 0
	for i := 0; i < len(dp); i++ {
		if dp[i] > max {
			max = dp[i]
		}
	}
	return max
}

func maxEnvelopes(envelopes [][]int) int {
	// 首先按宽度排序，然后问题就转化为了最长递增子数组
	sort.Slice(envelopes, func(i, j int) bool {
		return envelopes[i][0] < envelopes[j][0]
	})

	dp := make([]int, len(envelopes))
	dp[0] = 1
	for i := 1; i < len(dp); i++ {
		dp[i] = 1
		for j := i - 1; j >= 0; j-- {
			if envelopes[i][0] > envelopes[j][0] && envelopes[i][1] > envelopes[j][1] && dp[i] < dp[j]+1 {
				dp[i] = dp[j] + 1
			}
		}
	}

	max := 0
	for i := 0; i < len(dp); i++ {
		if dp[i] > max {
			max = dp[i]
		}
	}
	return max
}
func splitArraySameAverage(A []int) bool {
	sort.Ints(A)

	total := 0
	for i := 0; i < len(A); i++ {
		total += A[i]
	}

	average := float32(total) / float32(len(A))

	// pos 表示当前遍历到的位置，n 表示个数，total 表示总和
	var backtrack func(pos, n, total int) bool

	backtrack = func(pos, n, total int) bool {
		if float32(total)/float32(n) == average && n != len(A) {
			return true
		}
		if pos == len(A) {
			return false
		}

		return backtrack(pos+1, n+1, total+A[pos]) || backtrack(pos+1, n, total)
	}

	return backtrack(0, 0, 0)
}

// getIndexsOfMinInts get the indexs of all minium integer
func getIndexsOfMinInts(x ...int) []int {

	min := x[0]
	inds := []int{0}
	for i := 1; i < len(x); i++ {
		if x[i] < min {
			min = x[i]
			inds = []int{i}
		} else if x[i] == min {
			inds = append(inds, i)
		}
	}
	return inds
}

func sort3Ints(nums []int) {
	if nums[0] > nums[1] {
		nums[0], nums[1] = nums[1], nums[0]
	}
	if nums[1] > nums[2] {
		nums[1], nums[2] = nums[2], nums[1]
	}
}
func nthUglyNumber(n int, a int, b int, c int) int {
	var ret int
	ori := []int{a, b, c}
	nums := []int{a, b, c}

	for n > 0 {
		inds := getIndexsOfMinInts(nums...)
		ret = nums[inds[0]]
		for _, v := range inds {
			nums[v] += ori[v]
		}
		n--
	}

	return ret
}

func plusOne(digits []int) []int {
	ind := len(digits) - 1
	for ind >= 0 && digits[ind] == 9 {
		digits[ind] = 0
		ind--
	}
	if ind == -1 {
		digits = append([]int{1}, digits...)
		return digits
	}

	digits[ind]++

	return digits
}

func setZeroes(matrix [][]int) {
	rows := make([]int, 0)
	cols := make([]int, 0)

	for i := 0; i < len(matrix); i++ {
		for k := 0; k < len(matrix[0]); k++ {
			if matrix[i][k] == 0 {
				rows = append(rows, i)
				cols = append(cols, k)
			}
		}
	}

	for _, v := range rows {
		for i := 0; i < len(matrix[0]); i++ {
			matrix[v][i] = 0
		}
	}

	for _, v := range cols {
		for i := 0; i < len(matrix); i++ {
			matrix[i][v] = 0
		}
	}

}

func numDecodings(s string) int {

	if s[0] == '0' {
		return 0
	}
	// 应该使用动态规划,dp[i]表示以s[i-1]结尾的解码方式的个数
	dp := make([]int, len(s)+1)
	dp[0] = 1
	dp[1] = 1

	for i := 1; i < len(s); i++ {
		if s[i] == '0' {
			if s[i-1] != '1' && s[i-1] != '2' {
				return 0
			}
			dp[i+1] = dp[i-1]
		} else if s[i-1] == '0' {
			dp[i+1] = dp[i]
		} else {
			v, _ := strconv.Atoi(s[i-1 : i+1])
			if v > 26 {
				dp[i+1] = dp[i]
			} else {
				dp[i+1] = dp[i] + dp[i-1]
			}
		}
	}

	return dp[len(s)]
}

// 按照先序遍历的方式将二叉树转化为链表形式
func flatten(root *TreeNode) {
	if root == nil {
		return
	}
	pre := &TreeNode{} // 相当于头结点
	lst := list.New()
	lst.PushFront(root)

	for lst.Len() > 0 {
		node := lst.Front().Value.(*TreeNode)
		lst.Remove(lst.Front())

		if node.Right != nil {
			lst.PushFront(node.Right)
		}
		if node.Left != nil {
			lst.PushFront(node.Left)
		}

		node.Left = nil
		pre.Right = node
		pre = node
	}
}
func smallestSubsequence(s string) string {
	// 使用栈，当字符s[i]小于栈顶字符，并且s[i]之后还有和栈顶相同的字符，则弹出栈顶元素，压入s[i]
	contain := make([]int, 26)
	count := make([]int, 26)
	for i := 0; i < len(s); i++ {
		ind := s[i] - 'a'
		count[ind]++
	}

	lst := list.New()
	lst.PushBack(byte('a')) // 哨兵
	for i := 0; i < len(s); i++ {
		count[s[i]-'a']--
		if contain[s[i]-'a'] != 0 {
			continue
		}
		top := lst.Back().Value.(byte)
		for s[i] < top && count[top-'a'] > 0 {
			lst.Remove(lst.Back())
			contain[top-'a'] = 0
			top = lst.Back().Value.(byte)
		}
		lst.PushBack(s[i])
		contain[s[i]-'a']++
	}

	ret := make([]byte, lst.Len()-1)
	for i := len(ret) - 1; i >= 0; i-- {
		c := lst.Back().Value.(byte)
		ret[i] = c
		lst.Remove(lst.Back())
	}
	return string(ret)
}

func nextGreaterElements(nums []int) []int {
	ret := make([]int, len(nums))
	for i := 0; i < len(ret); i++ {
		ret[i] = -1
	}

	// 其中存储的是数组下标
	mono := make([]int, 0)

	for i := 0; i < len(nums)*2; i++ {
		ind := i % len(nums)

		for len(mono) != 0 && nums[mono[len(mono)-1]] < nums[ind] {
			ret[mono[len(mono)-1]] = nums[ind]
			mono = mono[0 : len(mono)-1]
		}
		mono = append(mono, ind)
	}

	return ret
}

func constructMaximumBinaryTree(nums []int) *TreeNode {
	if len(nums) == 0 {
		return nil
	}
	max, ind := 0, 0
	for i := 0; i < len(nums); i++ {
		if nums[i] > max {
			max = nums[i]
			ind = i
		}
	}

	root := &TreeNode{
		Val:   max,
		Left:  constructMaximumBinaryTree(nums[:ind]),
		Right: constructMaximumBinaryTree(nums[ind+1:]),
	}
	return root
}

func isPowerOfTwo(n int) bool {
	if n < 0 {
		return false
	}
	x := 1
	for i := 0; i < 32; i++ {
		if x == n {
			return true
		}
		x = x << 1
	}
	return false
}

func prisonAfterNDays(cells []int, n int) []int {
	var c uint8
	for i := 0; i < len(cells); i++ {
		c = c<<1 + uint8(cells[i])
	}

	count := make([]int, 256)
	for i := 0; i < len(count); i++ {
		count[i] = -1
	}
	count[c] = 0
	// 变化的过程会很快形成一个周期
	for i := 1; i <= n; i++ {
		left := c >> 1
		right := c << 1
		c = ^(left ^ right)
		c = c & 0xfe // 最后一位清零
		c = c & 0x7f // 第一位清零

		if count[c] != -1 {
			// 这时候周期就是i-count[c]，然后令n等于 (n-i)%(i-count[c])，i等于0，，直接跳到最后一个循环
			n = (n - i) % (i - count[c])
			for i := 1; i <= n; i++ {
				left := c >> 1
				right := c << 1
				c = ^(left ^ right)
				c = c & 0xfe // 最后一位清零
				c = c & 0x7f // 第一位清零
			}
			break
		} else {
			count[c] = i
		}
	}

	for i := len(cells) - 1; i >= 0; i-- {
		cells[i] = int(c & 0x01)
		c = c >> 1
	}
	return cells
}

// 给每一个节点一个编号（满二叉树中，按照从上到下从左到右编号）
// 如果当前节点的编号和遍历的节点个数不匹配，说明不是完全二叉树
func isCompleteTree(root *TreeNode) bool {

	count := 0

	lst := list.New()

	root.Val = 0 // 节点的val用来保存层数
	lst.PushBack(root)

	for lst.Len() > 0 {
		node := lst.Front().Value.(*TreeNode)
		lst.Remove(lst.Front())
		if node.Val != count {
			return false
		}
		count++
		if node.Left != nil {
			node.Left.Val = 2*node.Val + 1
			lst.PushBack(node.Left)
		}
		if node.Right != nil {
			node.Right.Val = 2*node.Val + 2
			lst.PushBack(node.Right)
		}
	}
	return true
}

func countPairs(root *TreeNode, distance int) int {

	ret := 0
	// 遍历函数，返回当前节点node的所有子节点中的叶子节点到当前节点的距离和个数 map[距离]个数
	// 如果node是叶子节点，则返回{0:1},距离为0，个数为1
	var dfs func(node *TreeNode) map[int]int

	dfs = func(node *TreeNode) map[int]int {
		m := make(map[int]int)
		if node == nil {
			return m
		}
		if node.Left == nil && node.Right == nil {
			m[0] = 1
			return m
		}

		left := dfs(node.Left)
		right := dfs(node.Right)

		// 对于left和right中的所有组合，如果相加距离小于给定值，则ret += vl * vr
		for kl, vl := range left {
			for kr, vr := range right {
				if kl+kr+2 <= distance {
					ret += vl * vr
				}
			}
		}

		for k, v := range left {
			// 距离过远的节点之后就不可能组成节点对了，所以可以直接不返回
			if k+2 <= distance {
				m[k+1] += v
			}
		}
		for k, v := range right {
			if k+2 <= distance {
				m[k+1] += v
			}
		}
		return m
	}

	dfs(root)
	return ret
}

// 计算sqrt（2） 精确到小数点后十位
func sqrt2() float64 {
	var low, high float64 = 0, 2

	mid := (low + high) / 2

	for math.Abs(mid*mid-2) > 1e-10 {
		if mid*mid > 2 {
			high = mid
		} else {
			low = mid
		}
		mid = (low + high) / 2
	}
	return float64(int64(math.Round(mid*1e10))) / 1e10
}

func findSingleNum(nums []int) int {
	var ret int

	for i := 0; i < len(nums); i++ {
		ret = ret ^ nums[i]
	}
	return ret
}

func findTwoSingalNum(nums []int) (int, int) {
	var res int
	for i := 0; i < len(nums); i++ {
		res = res ^ nums[i]
	}

	// 如果res中第shift位1，则将nums中的数按照shift位为0还是1进行划分为nums1,nums2，
	// 然后返回findSingleNum(nums1)和findSingleNum(nums2)
	shift := 1
	for res&shift == 0 {
		shift = shift << 1
	}

	var nums1, nums2 []int

	for i := 0; i < len(nums); i++ {
		if nums[i]&shift == 1 {
			nums1 = append(nums1, nums[i])
		} else {
			nums2 = append(nums2, nums[i])
		}
	}
	return findSingleNum(nums1), findSingleNum(nums2)
}

func countAndSay(n int) string {
	if n == 1 {
		return "1"
	}

	str := countAndSay(n - 1)

	ret := make([]byte, 0)

	v := str[0]
	c := 1

	for i := 1; i < len(str); i++ {
		if str[i] == v {
			c++
		} else {
			ret = strconv.AppendInt(ret, int64(c), 10)
			ret = append(ret, v)

			v = str[i]
			c = 1
		}
	}
	ret = strconv.AppendInt(ret, int64(c), 10)
	ret = append(ret, v)

	return string(ret)
}

func firstMissingPositive(nums []int) int {

	n := len(nums)
	buf := make([]int, n+1)

	for i := 0; i < len(nums); i++ {
		if nums[i] >= 0 && nums[i] <= n {
			buf[nums[i]] = 1
		}
	}

	for i := 0; i < n+1; i++ {
		if buf[i] == 0 {
			return i
		}
	}
	return n + 1
}

func reverseBetween(head *ListNode, left int, right int) *ListNode {

	ret := &ListNode{
		Next: head,
	}

	start := ret

	for i := 0; i < left-1; i++ {
		start = start.Next
	}

	p := start.Next

	for i := 1; i < right-left+1; i++ {

		tmp := p.Next

		p.Next = p.Next.Next

		tmp.Next = start.Next
		start.Next = tmp

	}

	return ret.Next
}

func cuttingRope(n int) int {
	if n == 0 {
		return 0
	}
	if n <= 2 {
		return 1
	}
	// dp[i]表示长度为i的绳子的最大乘积
	dp := make([]int, n+1)

	dp[1] = 1
	dp[2] = 1
	for i := 3; i <= n; i++ {
		for j := 2; j < i; j++ {
			if dp[i-j]*j > dp[i] {
				dp[i] = dp[i-j] * j
			}

			if (i-j)*j > dp[i] {
				dp[i] = (i - j) * j
			}
		}
	}

	return dp[n]
}

// 最长递增路径，使用bfs + 动态规划解决
func longestIncreasingPath(matrix [][]int) int {
	m := len(matrix)
	if m == 0 {
		return 0
	}
	n := len(matrix[0])

	dp := make([][]int, m)
	for i := 0; i < m; i++ {
		dp[i] = make([]int, n)
	}

	var ret int

	var bfs func(matrix [][]int, x, y int)

	dirX := []int{-1, 0, 1, 0}
	dirY := []int{0, 1, 0, -1}
	bfs = func(matrix [][]int, x, y int) {

		if dp[x][y] != 0 {
			return
		}

		for i := 0; i < len(dirX); i++ {
			nx := x + dirX[i]
			ny := y + dirY[i]

			// 只往较大的方向扩展
			if nx >= 0 && nx < m && ny >= 0 && ny < n && matrix[x][y] < matrix[nx][ny] {
				bfs(matrix, nx, ny)
				if dp[x][y] < dp[nx][ny]+1 {
					dp[x][y] = dp[nx][ny] + 1

				}
			}
		}

		// 周围没有比他更小的了，设置为1
		if dp[x][y] == 0 {
			dp[x][y] = 1
		}
	}

	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			bfs(matrix, i, j)
			if dp[i][j] > ret {
				ret = dp[i][j]
			}
		}
	}

	return ret
}

func dominantIndex(nums []int) int {
	first, second := nums[0], 0
	ret := 0
	for i := 1; i < len(nums); i++ {
		if nums[i] > first {
			ret = i
			second = first
			first = nums[i]
		} else if nums[i] > second {
			second = nums[i]
		}
	}

	if first >= second*2 {
		return ret
	}
	return -1
}

// 返回线段到点的最短距离
//  竖直向上direction = true，水平向右 direction = false
func distanceFromPointToSegment(x0, y0, x1, y1, length float64, direction bool) float64 {
	if direction {
		if y0 < y1 || y0 > y1+length {
			return math.Min(
				math.Sqrt((x0-x1)*(x0-x1)+(y0-y1)*(y0-y1)),
				math.Sqrt((x0-x1)*(x0-x1)+(y0-y1-length)*(y0-y1-length)),
			)
		} else {
			return math.Abs(x0 - x1)
		}
	} else {
		if x0 < x1 || x0 > x1+length {
			return math.Min(
				math.Sqrt((x0-x1)*(x0-x1)+(y0-y1)*(y0-y1)),
				math.Sqrt((x0-x1-length)*(x0-x1-length)+(y0-y1)*(y0-y1)),
			)
		} else {
			return math.Abs(y0 - y1)
		}
	}
}
func checkOverlap(radius int, x_center int, y_center int, x1 int, y1 int, x2 int, y2 int) bool {

	fradius, fxc, fyc, fx1, fy1, fx2, fy2 := float64(radius), float64(x_center), float64(y_center),
		float64(x1), float64(y1), float64(x2), float64(y2)

	dl := distanceFromPointToSegment(fxc, fyc, fx1, fy1, fy2-fy1, true)
	dr := distanceFromPointToSegment(fxc, fyc, fx2, fy1, fy2-fy1, true)

	du := distanceFromPointToSegment(fxc, fyc, fx1, fy2, fx2-fx1, false)
	dd := distanceFromPointToSegment(fxc, fyc, fx1, fy1, fx2-fx1, false)

	if dl <= fradius || dr <= fradius || du <= fradius || dd <= fradius {
		return true
	}

	// 圆心在矩形里面
	if x_center >= x1 && x_center <= x2 && y_center >= y1 && y_center <= y2 {
		return true
	}

	return false
}

func combine(n int, k int) [][]int {

	ret := [][]int{}
	if k == 0 {
		return [][]int{{}}
	}
	if n == k {
		ret = append(ret, []int{})
		for i := 1; i <= n; i++ {
			ret[0] = append(ret[0], i)
		}
		return ret
	}

	pick := combine(n-1, k-1)
	nopick := combine(n-1, k)

	for i := 0; i < len(pick); i++ {
		pick[i] = append(pick[i], n)
	}

	ret = append(ret, pick...)
	ret = append(ret, nopick...)
	return ret
}

/**
 *
 * @param x int整型
 * @return int整型
 */
func sqrt(x int) int {
	low, high := 1, x
	for low < high-1 {
		mid := (low + high) / 2
		if mid*mid == x {
			return mid
		} else if mid*mid < x {
			low = mid
		} else {
			high = mid
		}
	}
	return low
}

func buildTree(preorder []int, inorder []int) *TreeNode {

	if len(preorder) == 0 {
		return nil
	}

	root := &TreeNode{
		Val: preorder[0],
	}
	if len(preorder) == 1 {
		return root
	}
	find := func(nums []int, target int) int {
		for i := 0; i < len(nums); i++ {
			if nums[i] == target {
				return i
			}
		}

		return -1
	}

	pos := find(inorder, preorder[0])

	root.Left = buildTree(preorder[1:1+pos], inorder[:pos])
	root.Right = buildTree(preorder[1+pos:], inorder[pos+1:])
	return root
}

// 仅仅保留不重复的元素
func deleteDuplicates(head *ListNode) *ListNode {
	ret := &ListNode{
		Next: head,
	}

	pre, p := ret, ret.Next

	for p != nil {
		for p.Val == p.Next.Val {
			p = p.Next
		}
		if p.Next != p {
			pre.Next = p.Next
		} else {
			pre = p
			p = p.Next
		}
	}

	return ret.Next
}

// 接雨水
// 对于位置i，它能接的雨水的最大数量取决于向左看（包括自己）的最大值lMax，向右看的最大值rMax，接的雨水量问 max(min(lMax, rMax) - height[i], 0)
func trap(height []int) int {
	top := make([]int, len(height))
	stack := []int{}

	for i := 0; i < len(height); i++ {
		for len(stack) > 0 && stack[len(stack)-1] < height[i] {
			stack = stack[:len(stack)-1]
		}
		stack = append(stack, height[i])
		top[i] = stack[0]
	}

	stack = []int{}

	for i := len(height) - 1; i >= 0; i-- {
		for len(stack) > 0 && stack[len(stack)-1] < height[i] {
			stack = stack[:len(stack)-1]
		}
		stack = append(stack, height[i])

		if stack[0] < top[i] {
			top[i] = stack[0]
		}
	}

	ret := 0
	for i := 0; i < len(height); i++ {
		ret += top[i] - height[i]
	}

	return ret
}

// 440. 字典序的第K小数字
// 解题方式，构建字典树，将数字一次插入字典树中，然后按顺序遍历到第k个
func findKthNumber(n int, k int) int {

	words := make([]*trieNode, 10)

	for i := 1; i <= n; i++ {
		str := strconv.Itoa(i)
		cur := words
		for j := 0; j < len(str); j++ {
			if cur[str[j]-'0'] == nil {
				cur[str[j]-'0'] = &trieNode{
					next: make([]*trieNode, 10),
				}
			}
			if j == len(str)-1 {
				cur[str[j]-'0'].val = i
				break
			}
			cur = cur[str[j]-'0'].next

		}
	}

	// 遍历字典树，返回第k个元素
	lst := list.New()
	for i := 0; i < len(words); i++ {
		if words[i] != nil {
			lst.PushBack(words[i])
		}
	}
	for lst.Len() > 0 {
		node := lst.Front().Value.(*trieNode)
		lst.Remove(lst.Front())
		if node.val != 0 {
			k--
		}
		if k == 0 {
			return node.val
		}

		for i := 0; i < len(node.next); i++ {
			if node.next[i] != nil {
				lst.PushBack(node.next[i])
			}
		}
	}

	return 0
}

type trieNode struct {
	next []*trieNode
	val  int
}

func reverseBits(num uint32) uint32 {
	var ret uint32

	for i := 0; i < 32; i++ {
		ret = (ret<<1 | (num & 1))
		num = num >> 1
	}

	return ret
}

func isMatch(s string, p string) bool {

	if len(s) == 0 {
		return emptiable(p)
	}

	if len(p) == 0 {
		return len(s) == 0
	}

	if len(p) == 1 {
		return len(s) == 1 && (s[0] == p[0] || p[0] == '.')
	}

	if p[1] != '*' {
		if s[0] == p[0] || p[0] == '.' {
			return isMatch(s[1:], p[1:])
		} else {
			return false
		}
	} else {
		for repeat := 0; repeat <= len(s) && (repeat == 0 || s[repeat-1] == p[0] || p[0] == '.'); repeat++ {
			if isMatch(s[repeat:], p[2:]) {
				return true
			}
		}
		return false
	}
}

func emptiable(p string) bool {
	return p == "" || (len(p) >= 2 && p[1] == '*' && emptiable(p[2:]))
}

func searchMatrix(matrix [][]int, target int) bool {
	// 首先以行为单位，进行二分查找确定某一行，然后在那一行上进行二分查找确定一个数
	m, n := len(matrix), len(matrix[0])

	rl, rh := 0, m-1

	for rl < rh {
		mid := (rl + rh) / 2
		if matrix[mid][n-1] == target {
			return true
		}
		if matrix[mid][n-1] > target {
			rh = mid
		} else {
			rl = mid + 1
		}
	}

	// 现在target一定在rh那一行上

	low, high := 0, n-1

	for low <= high {
		mid := (low + high) / 2
		if matrix[rh][mid] == target {
			return true
		}
		if matrix[rh][mid] > target {
			high = mid - 1
		} else {
			low = mid + 1
		}
	}

	return false
}

func subsetsWithDup(nums []int) [][]int {
	sort.Ints(nums)

	var ret [][]int

	var dfs func(pos int, output []int)

	dfs = func(pos int, output []int) {

		if pos == len(nums) {
			tmp := make([]int, len(output))
			copy(tmp, output)
			ret = append(ret, tmp)
		} else {
			// 包含当前数字
			output = append(output, nums[pos])
			dfs(pos+1, output)
			output = output[:len(output)-1]

			// 不包含当前数字
			pos++
			for pos < len(nums) && nums[pos] == nums[pos-1] {
				pos++
			}
			dfs(pos, output)
		}
	}

	dfs(0, []int{})

	return ret
}

func clumsy(N int) int {
	return 0
}

func longestCommonSubsequence(text1 string, text2 string) int {
	m := len(text1)
	if m == 0 {
		return 0
	}
	n := len(text2)

	dp := make([][]int, m+1)

	for i := 0; i < m+1; i++ {
		dp[i] = make([]int, n+1)
	}

	for i := 1; i < m+1; i++ {
		for j := 1; j < n+1; j++ {
			if text1[i-1] == text2[j-1] {
				dp[i][j] = dp[i-1][j-1] + 1
			} else if dp[i][j-1] > dp[i-1][j] {
				dp[i][j] = dp[i][j-1]
			} else {
				dp[i][j] = dp[i-1][j]
			}
		}
	}

	return dp[m][n]
}

func removeDuplicates(nums []int) int {

	count := 1
	val := nums[0]
	for i := 1; i < len(nums); i++ {
		if nums[i] == val {
			count++
			if count > 2 {
				nums[i] -= count
			}
		} else {
			count = 1
			val = nums[i]
		}
	}

	p := 1
	for i := 1; i < len(nums); i++ {
		if nums[i] >= nums[i-1] {
			nums[p] = nums[i]
			p++
		}
	}

	return p
}

func findLeastNumOfUniqueInts(arr []int, k int) int {
	m := make(map[int]int)
	for i := 0; i < len(arr); i++ {
		m[arr[i]]++
	}

	sort.Slice(arr, func(i, j int) bool {
		if m[arr[i]] != m[arr[j]] {
			return m[arr[i]] < m[arr[j]]
		} else {
			return arr[i] < arr[j]
		}
	})

	count := 0
	for i := k; i < len(arr); i++ {
		if i == k || arr[i] != arr[i-1] {
			count++
		}
	}

	return count
}

func search(nums []int, target int) bool {
	// 找到分界点，然后在左右两部分分别二分查找？
	ind := sort.Search(len(nums), func(i int) bool {
		if i == 0 {
			return false
		}
		return nums[i] <= nums[0]
	})

	var i int
	left := nums[:ind]
	right := nums[ind:]

	i = sort.SearchInts(left, target)
	if i != len(left) && left[i] == target {
		return true
	}

	i = sort.SearchInts(right, target)
	if i != len(right) && right[i] == target {
		return true
	}
	return false
}

func findMin(nums []int) int {
	// 右半部分的最大值要小于等于左半部分的最小值
	left, right := 0, len(nums)-1
	if nums[left] < nums[right] { // 说明没有旋转
		return nums[0]
	}
	// 第一步：在右半部分找到第一个小于nums[left]的数
	for left < right && nums[left] == nums[right] && nums[right] >= nums[right-1] {
		right--
	}

	for left < right-1 {
		mid := (left + right) / 2
		if nums[mid] >= nums[left] { // 说明在左半部分
			left = mid
		} else { // 在右半部分
			right = mid
		}
	}

	return nums[right]
}

func scoreOfParentheses(S string) int {
	// 设计一个栈，
	// 当遇到左括号时，压入0
	// 遇到右括号时，如果栈顶是0，则弹出栈顶，然后栈顶元素加一
	// 如果栈顶不是0，设为x，则弹出栈顶，然后栈顶元素加 2x

	stack := make([]int, 1) // 遍历完成后，剩下的元素就是最终的结果

	for _, c := range S {
		if c == '(' {
			stack = append(stack, 0)
		} else {
			x := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			if x == 0 {
				stack[len(stack)-1] += 1
			} else {
				stack[len(stack)-1] += x + x
			}
		}
	}
	return stack[0]
}

func findTheWinner(n int, k int) int {
	if k == 1 {
		return n
	}
	next := make([]int, n)
	for i := 0; i < len(next); i++ {
		next[i] = (i + 1) % n
	}

	p := 0
	for i := 0; i < n-1; i++ {
		for j := 0; j < k-2; j++ {
			p = next[p]
		}
		next[p] = next[next[p]]
		p = next[p]
	}

	return next[p] + 1
}

func minSideJumps(obstacles []int) int {
	// 贪心选择策略，每次侧跳选择障碍物距离最远的那条路

	var getNext func(r1, r2, i int) (int, int) = func(r1, r2, i int) (int, int) {
		l1, l2 := 0, 0

		for l1 = 0; l1 < len(obstacles)-i && obstacles[i+l1] != r1; l1++ {

		}
		for l2 = 0; l2 < len(obstacles)-i && obstacles[i+l2] != r2; l2++ {

		}
		if l1 > l2 {
			return r1, i + l1
		}
		return r2, i + l2
	}

	ret := 0
	cur := 2
	for i := 0; i < len(obstacles)-1; i++ {
		if obstacles[i+1] == cur { // 遇到障碍物
			if obstacles[i] != 0 {
				cur = 6 - cur - obstacles[i]
			} else if obstacles[i+1] == 1 {
				cur, i = getNext(2, 3, i)
			} else if obstacles[i+1] == 2 {
				cur, i = getNext(1, 3, i)
			} else {
				cur, i = getNext(1, 2, i)
			}
			ret++
		}
	}
	return ret
}

func largestNumber(nums []int) string {
	strs := make([]string, len(nums))

	var ret strings.Builder

	for i := 0; i < len(nums); i++ {
		strs[i] = strconv.FormatInt(int64(nums[i]), 10)
	}

	sort.Slice(strs, func(i, j int) bool {
		return strs[i]+strs[j] < strs[j]+strs[i]
	})

	for i := len(strs) - 1; i >= 0; i-- {
		ret.WriteString(strs[i])
	}

	str := strings.TrimLeft(ret.String(), "0")

	if len(str) == 0 {
		return "0"
	}
	return str
}

func minDiffInBST(root *TreeNode) int {

	ret := math.MaxInt32
	pre := math.MinInt32

	var inOrder func(node *TreeNode)

	inOrder = func(node *TreeNode) {
		if node == nil {
			return
		}

		inOrder(node.Left)

		if node.Val-pre < ret {
			ret = node.Val - pre
		}
		pre = node.Val

		inOrder(node.Right)
	}

	inOrder(root)

	return ret
}

func oddEvenList(head *ListNode) *ListNode {
	odd := &ListNode{}
	even := &ListNode{}

	p, po, pe := head, odd, even
	i := 0
	for p != nil {
		if i%2 == 0 {
			pe.Next = p
			pe = pe.Next
		} else {
			po.Next = p
			po = po.Next
		}
		i++
		p = p.Next
	}

	pe.Next = odd.Next
	po.Next = nil
	return even.Next
}

// 找递增子序列
func findSubsequences(nums []int) [][]int {
	// dp[i] 表示以 nums[i] 结尾的递增子序列
	dp := make([][][]int, len(nums))

	dp[0] = [][]int{{nums[0]}}
	for i := 1; i < len(nums); i++ {
		for j := i - 1; j >= 0; j-- {
			if nums[i] >= nums[j] {
				preLen := len(dp[i])
				dp[i] = append(dp[i], dp[j]...)
				for k := preLen; k < len(dp[i]); k++ {
					dp[i][k] = append(dp[i][k], nums[i])
				}
			}
		}
	}

	ret := make([][]int, 0)

	for i := 0; i < len(dp); i++ {
		ret = append(ret, dp[i]...)
	}

	return ret[1:]
}

func rob(nums []int) int {
	if len(nums) == 1 {
		return nums[0]
	}

	var noCircleRob = func(nums []int) int {

		if len(nums) == 0 {
			return 0
		}
		if len(nums) == 1 {
			return nums[1]
		}
		dp := make([]int, len(nums))

		dp[0] = nums[0]
		if nums[0] > nums[1] {
			dp[1] = nums[0]
		} else {
			dp[1] = nums[1]
		}

		for i := 2; i < len(nums); i++ {
			if dp[i-1] < dp[i-2]+nums[i] {
				dp[i] = dp[i-2] + nums[i]
			} else {
				dp[i] = dp[i-1]
			}
		}

		return dp[len(dp)-1]
	}

	A := noCircleRob(nums[:len(nums)-1])
	B := noCircleRob(nums[1:])

	if A > B {
		return A
	}
	return B
}

func isScramble(s1 string, s2 string) bool {
	// dp1[i,j,k]表示s1[i,i+k] 和s2[j, j+k]是不是scramble
	type threeInts struct {
		i, j, k int
	}
	m := make(map[threeInts]bool)

	var helper func(i, j, k int) bool
	helper = func(i, j, k int) bool {

		if val, ok := m[threeInts{i, j, k}]; ok {
			return val
		}
		if k == 1 {
			m[threeInts{i, j, k}] = s1[i] == s2[i]
			return s1[i] == s2[j]
		}

		for ind := 1; ind < k; ind++ {
			if (helper(i, j, ind) && helper(i+ind, j+ind, k-ind)) ||
				(helper(i, j+k-ind, ind) && helper(i+ind, j, k-ind)) {
				m[threeInts{i, j, k}] = true
				return true
			}
		}
		m[threeInts{i, j, k}] = false
		return false
	}
	return helper(0, 0, len(s1))
}

func numFactoredBinaryTrees(arr []int) int {
	m := make(map[int]int)
	for i := range arr {
		m[arr[i]] = i
	}
	// dp[i] 表示以arr[i]作为根节点的树的个数
	dp := make([]int, len(arr))
	for i := range dp {
		dp[i] = 1
	}
	for i := 1; i < len(arr); i++ {
		for j := 0; j < i; j++ {
			if arr[i]%arr[j] == 0 {
				ind, ok := m[arr[i]/arr[j]]
				if ok {
					dp[i] += (dp[j] * dp[ind]) % Mod
				}
			}
		}
	}
	ret := 0
	for i := range dp {
		ret = (ret + dp[i]) % Mod
	}
	return ret
}

func maxSumSubmatrix(matrix [][]int, k int) int {
	m := len(matrix)
	n := len(matrix[0])
	subSum := make([][]int, m+1)
	for i := 0; i < len(subSum); i++ {
		subSum[i] = make([]int, n+1)
	}

	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			subSum[i][j] = subSum[i-1][j] + subSum[i][j-1] + matrix[i-1][j-1] - subSum[i-1][j-1]
		}
	}

	ret := math.MinInt64
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			for p := i; p <= m; p++ {
				for q := j; q <= n; q++ {
					val := subSum[p][q] - subSum[i-1][q] - subSum[p][j-1] + subSum[i-1][j-1]
					if val <= k && val > ret {
						ret = val
					}
				}
			}
		}
	}

	return ret
}
func combinationSum4(nums []int, target int) int {
	dp := make([]int, target+1)
	dp[0] = 1
	for i := 1; i <= target; i++ {
		for _, num := range nums {
			if num <= i {
				dp[i] += dp[i-num]
			}
		}
	}
	return dp[target]
}

// 可以采用二分查找的方式，小于最小值时，没有办法运送完毕，大于最小值时，可以运送完毕，中间的分界点就是需要的结果
func shipWithinDays(weights []int, D int) int {
	var low, high int
	var max int
	for i := 0; i < len(weights); i++ {
		high += weights[i]
		if weights[i] > max {
			max = weights[i]
		}
	}
	low = max - 1
	var canShip func(cap int) bool = func(cap int) bool {
		cur, count := 0, 0
		for i := 0; i < len(weights); i++ {
			if cur+weights[i] <= cap {
				cur += weights[i]
			} else {
				count++
				cur = weights[i]
			}
		}
		return count+1 <= D
	}

	for low < high-1 {
		mid := (low + high) / 2
		if canShip(mid) {
			high = mid
		} else {
			low = mid
		}
	}

	return high
}
func smallestGoodBase(n string) string {

	return ""
}

// 层序遍历，计算每一层的宽度
func widthOfBinaryTree(root *TreeNode) int {
	ret := 0
	st1, st2 := []*TreeNode{root}, []*TreeNode{}
	root.Val = 0
	for len(st1) > 0 {
		if st1[len(st1)-1].Val-st1[0].Val > ret {
			ret = st1[len(st1)-1].Val - st1[0].Val
		}
		for len(st1) != 0 {
			node := st1[0]
			st1 = st1[1:]
			if node.Left != nil {
				node.Left.Val = node.Val*2 + 1
				st2 = append(st2, node.Left)
			}
			if node.Right != nil {
				node.Right.Val = node.Val*2 + 2
				st2 = append(st2, node.Right)
			}
		}

		st2, st1 = st1, st2
	}
	return ret
}

func judgeSquareSum(c int) bool {
	left, right := 0, int(math.Sqrt(float64(c)))
	for left <= right {
		sum := left*left + right*right
		if sum == c {
			return true
		} else if sum > c {
			right--
		} else {
			left++
		}
	}
	return false
}

// 采用广度优先遍历的方式
func allPathsSourceTarget(graph [][]int) [][]int {
	var ret [][]int
	n := len(graph)

	var path, visited []int
	visited = make([]int, n)

	var bfs func(ind int)
	bfs = func(ind int) {
		if ind == n-1 {
			tmp := make([]int, len(path))
			copy(tmp, path)
			ret = append(ret, tmp)
			return
		}
		for _, v := range graph[ind] {
			if visited[v] == 0 {
				path = append(path, v)
				visited[v] = 1
				bfs(v)
				path = path[:len(path)-1]
				visited[v] = 0
			}
		}
	}

	path = append(path, 0)
	visited[0] = 1

	bfs(0)
	return ret
}

func busiestServers(k int, arrival []int, load []int) []int {

	type server struct {
		t      int
		taskID int
		count  int
		id     int
	}
	sers := make([]server, k)
	for i := 0; i < k; i++ {
		sers[i].taskID = -1
		sers[i].id = i
	}
	for i := 0; i < len(arrival); i++ {
		for j := 0; j < k; j++ {
			ind := (i%k + j) % k
			if sers[ind].taskID == -1 || (arrival[i]-sers[ind].t >= load[sers[ind].taskID]) {
				sers[ind].count++
				sers[ind].t = arrival[i]
				sers[ind].taskID = i
				break
			}
		}
	}

	sort.Slice(sers, func(i, j int) bool {
		return sers[i].count > sers[j].count
	})

	var ret []int
	for i := 0; i < k; i++ {
		if i != 0 && sers[i].count != sers[i-1].count {
			break
		}
		ret = append(ret, sers[i].id)
	}
	return ret
}

// 最多走len(stones)步
func canCross(stones []int) bool {
	// 0表示未知，1 表示能走到终点，2 表示不能
	dp := make([][]int, len(stones))
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]int, len(stones))
	}
	var jump func(ind, k int) bool
	jump = func(ind, k int) bool {
		if dp[ind][k] != 0 {
			return dp[ind][k] == 1
		}
		if ind == len(stones)-1 {
			return true
		}
		i := 1
		var A, B, C bool
		for ind+i < len(stones) && stones[ind+i]-stones[ind] <= k+1 {
			if stones[ind+i]-stones[ind] == k-1 {
				A = jump(ind+i, k-1)
			} else if stones[ind+i]-stones[ind] == k {
				B = jump(ind+i, k)
			} else if stones[ind+i]-stones[ind] == k+1 {
				C = jump(ind+i, k+1)
			}
			i++
		}
		if A || B || C {
			dp[ind][k] = 1
		} else {
			dp[ind][k] = 2
		}
		return A || B || C
	}
	if stones[1]-stones[0] != 1 {
		return false
	}
	return jump(1, 1)
}
func decode(encoded []int, first int) []int {
	ret := make([]int, len(encoded)+1)
	ret[0] = first
	for i := 1; i < len(ret); i++ {
		ret[i] = ret[i-1] ^ encoded[i-1]
	}
	return ret
}

// 采用二分的方式,假设最小值为k，那么小于k的时候无法完成工作，大于k的时候可以完成工作
func minimumTimeRequired(jobs []int, k int) int {
	var canFinish = func(t int) bool {
		if jobs[len(jobs)-1] > t {
			return false
		}
		done := make([]bool, len(jobs))
		count := 0 // 分配出去的任务数
		for i := 0; i < k; i++ {
			if count == len(jobs) {
				return true
			}
			// 对于每一个人，需要找到最接近t的任务组合
			cur := 0 // 当前分配的任务量
			for i := len(jobs) - 1; i >= 0; i-- {
				if !done[i] && cur+jobs[i] <= t {
					done[i] = true
					cur = cur + jobs[i]
					count++
				}
			}
		}
		return count == len(jobs)
	}

	sort.Ints(jobs)

	sum := 0
	for i := 0; i < len(jobs); i++ {
		sum += jobs[i]
	}
	low, high := 0, sum

	for low < high-1 {
		mid := (low + high) / 2
		if canFinish(mid) {
			high = mid
		} else {
			low = mid
		}
	}
	return high
}

func main() {
	jobs := []int{6518448, 8819833, 7991995, 7454298, 2087579, 380625, 4031400, 2905811, 4901241, 8480231, 7750692, 3544254}
	fmt.Println(minimumTimeRequired(jobs, 4))
}
