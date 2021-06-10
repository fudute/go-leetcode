package algorithm

var m map[int]*Elem

func init() {
	m = make(map[int]*Elem)
}

type Set struct {
	head *Elem
	tail *Elem
}

type Elem struct {
	s    *Set
	val  int
	next *Elem
}

func MakeSet(x int) *Set {
	e := &Elem{}
	s := &Set{}
	e.s = s
	e.val = x
	e.next = nil
	m[x] = e

	s.head = e
	s.tail = e
	return s
}

func Union(x, y int) *Set {
	sx, sy := FindSet(x), FindSet(y)
	if sx != sy {
		p := sy.head
		for p != nil {
			p.s = sx
			sx.tail.next = p
			sx.tail = sx.tail.next
			p = p.next
		}
	}
	return sx
}

func FindSet(x int) *Set {
	return m[x].s
}
