package main

import (
	"container/list"
	"fmt"
	"math/bits"
	"strings"
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

func quickSort(nums []int) {
	if len(nums) == 1 || len(nums) == 0 {
		return
	}
	cmp := nums[0]
	low, high := 0, len(nums)-1
	for low < high {
		for low < high && nums[high] >= cmp {
			high--
		}
		nums[low] = nums[high]
		for low < high && nums[low] <= cmp {
			low++
		}
		nums[high] = nums[low]
	}
	nums[low] = cmp
	quickSort(nums[0:low])
	quickSort(nums[low+1:])
	return
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

func printList(head *ListNode) {
	for head != nil {
		fmt.Printf("%d ", head.Val)
		head = head.Next
	}
	fmt.Println()
}

func reverse(s []int) []int {
	for i, j := 0, len(s)-1; i < j; i, j = i+1, j-1 {
		s[i], s[j] = s[j], s[i]
	}
	return s
}

type intPair struct {
	val   int // val
	index int // index
}

// 从后遍历，然后大到下压入栈——单调栈
func dailyTemperatures(T []int) []int {
	ret := make([]int, len(T))
	s := make([]intPair, 0)
	for i := len(T) - 1; i >= 0; i-- {
		tmp := len(s) - 1
		for tmp >= 0 && s[tmp].val <= T[i] {
			tmp--
		}
		s = s[0 : tmp+1]
		if tmp == -1 {
			ret[i] = 0
		} else {
			ret[i] = s[tmp].index - i
		}
		s = append(s, intPair{val: T[i], index: i})
	}
	return ret
}
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

func min(lhs, rhs int) int {
	if lhs < rhs {
		return lhs
	}
	return rhs
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

	for i, num := range nums {
		// 把num从 nums 拿出去 得到tmp
		tmp := make([]int, len(nums)-1)
		copy(tmp[0:], nums[0:i])
		copy(tmp[i:], nums[i+1:])

		// sub 是把num 拿出去后，数组中剩余数据的全排列
		sub := permute(tmp)
		for _, s := range sub {
			res = append(res, append(s, num))
		}
	}
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

// IntMax int的最大值
const IntMax = int(^uint(0) >> 1)

// IntMin int的最小值
const IntMin = ^IntMax

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
		l1, l2, r1, r2 := IntMin, IntMin, IntMax, IntMax

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

func main() {
	nums1 := []int{2, 3}
	nums2 := []int{1}

	fmt.Println(findMedianSortedArrays(nums1, nums2))
}
