package utils

import (
	"math/rand"
	"strings"
)

// RandomString 随机生成长度介于low，high之间的字符串
func RandomString(low, high int) string {
	var sb strings.Builder
	ln := rand.Int()%(high-low) + low
	for i := 0; i < ln; i++ {
		c := 'a' + rand.Int()%26
		sb.WriteByte(byte(c))
	}
	return sb.String()
}
