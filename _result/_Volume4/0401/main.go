package main

import (
	"bufio"
	"fmt"
	"io/ioutil"
	"math"
	"math/bits"
	"os"
	"sort"
	"strconv"
	"strings"
)

var sc = bufio.NewScanner(os.Stdin)
var wtr = bufio.NewWriter(os.Stdout)

func main() {

	defer flush()

	gd := make([][]int, 201)
	for i := 0; i < 201; i++ {
		gd[i] = is(201, -1)
	}
	gd[0][0] = 0
	var cgd func(w, b int) int
	cgd = func(w, b int) int {
		if gd[w][b] != -1 {
			return gd[w][b]
		}
		m := make([]bool, 300)
		if w > 0 {
			m[cgd(w-1, b)] = true
		}
		for i := 1; i <= w; i++ {
			if b-i >= 0 {
				m[cgd(w, b-i)] = true
			}
		}
		if b > 0 {
			m[cgd(w+1, b-1)] = true
		}
		for i := 0; i < 301; i++ {
			if m[i] != true {
				gd[w][b] = i
				break
			}
		}
		return gd[w][b]
	}
	for i := 0; i < 201; i++ {
		for j := 0; j < 101; j++ {
			if i+j > 200 {
				continue
			}
			cgd(i, j)
		}
	}

	o := 0
	n := ni()
	ws, bs := ni2s(n)
	t := 0
	for i := 0; i < n; i++ {
		t ^= gd[ws[i]][bs[i]]
	}
	if t == 0 {
		o++
	}
	out(o)
}

// ==================================================
// init
// ==================================================

const inf = math.MaxInt64
const mod1000000007 = 1000000007
const mod998244353 = 998244353
const mod = mod1000000007

func init() {
	sc.Buffer([]byte{}, math.MaxInt64)
	sc.Split(bufio.ScanWords)
	if len(os.Args) > 1 && os.Args[1] == "i" {
		b, e := ioutil.ReadFile("./input")
		if e != nil {
			panic(e)
		}
		sc = bufio.NewScanner(strings.NewReader(strings.Replace(string(b), " ", "\n", -1)))
	}
}

// ==================================================
// io
// ==================================================

func ni() int {
	sc.Scan()
	i, e := strconv.Atoi(sc.Text())
	if e != nil {
		panic(e)
	}
	return i
}

func ni2() (int, int) {
	return ni(), ni()
}

func ni3() (int, int, int) {
	return ni(), ni(), ni()
}

func ni4() (int, int, int, int) {
	return ni(), ni(), ni(), ni()
}

func nis(n int) []int {
	a := make([]int, n)
	for i := 0; i < n; i++ {
		a[i] = ni()
	}
	return a
}

func ni2s(n int) ([]int, []int) {
	a := make([]int, n)
	b := make([]int, n)
	for i := 0; i < n; i++ {
		a[i], b[i] = ni2()
	}
	return a, b
}

func ni3s(n int) ([]int, []int, []int) {
	a := make([]int, n)
	b := make([]int, n)
	c := make([]int, n)
	for i := 0; i < n; i++ {
		a[i], b[i], c[i] = ni3()
	}
	return a, b, c
}

func ni4s(n int) ([]int, []int, []int, []int) {
	a := make([]int, n)
	b := make([]int, n)
	c := make([]int, n)
	d := make([]int, n)
	for i := 0; i < n; i++ {
		a[i], b[i], c[i], d[i] = ni4()
	}
	return a, b, c, d
}

func ni2a(n int) [][2]int {
	a := make([][2]int, n)
	for i := 0; i < n; i++ {
		a[i][0], a[i][1] = ni2()
	}
	return a
}

func nf() float64 {
	sc.Scan()
	f, e := strconv.ParseFloat(sc.Text(), 64)
	if e != nil {
		panic(e)
	}
	return f
}

func ns() string {
	sc.Scan()
	return sc.Text()
}

func out(v ...interface{}) {
	_, e := fmt.Fprintln(wtr, v...)
	if e != nil {
		panic(e)
	}
}

func outwoln(v ...interface{}) {
	_, e := fmt.Fprint(wtr, v...)
	if e != nil {
		panic(e)
	}
}

func outis(sl []int) {
	r := make([]string, len(sl))
	for i, v := range sl {
		r[i] = itoa(v)
	}
	out(strings.Join(r, " "))
}

func flush() {
	e := wtr.Flush()
	if e != nil {
		panic(e)
	}
}

func nftoi(decimalLen int) int {
	sc.Scan()
	s := sc.Text()

	r := 0
	minus := strings.Split(s, "-")
	isMinus := false
	if len(minus) == 2 {
		s = minus[1]
		isMinus = true
	}

	t := strings.Split(s, ".")
	i := atoi(t[0])
	r += i * pow(10, decimalLen)
	if len(t) > 1 {
		i = atoi(t[1])
		i *= pow(10, decimalLen-len(t[1]))
		r += i
	}
	if isMinus {
		return -r
	}
	return r
}

func atoi(s string) int {
	i, e := strconv.Atoi(s)
	if e != nil {
		panic(e)
	}
	return i
}

func itoa(i int) string {
	return strconv.Itoa(i)
}

func btoi(b byte) int {
	return atoi(string(b))
}

// ==================================================
// num
// ==================================================

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func abs(a int) int {
	if a > 0 {
		return a
	}
	return -a
}

func pow(a, b int) int {
	return int(math.Pow(float64(a), float64(b)))
}

func pow2(a int) int {
	return int(math.Pow(2, float64(a)))
}

func pow10(a int) int {
	return int(math.Pow(10, float64(a)))
}

func sqrt(i int) int {
	return int(math.Sqrt(float64(i)))
}

func ch(cond bool, ok, ng int) int {
	if cond {
		return ok
	}
	return ng
}

func mul(a, b int) (int, int) {
	if a < 0 {
		a, b = -a, -b
	}
	if a == 0 || b == 0 {
		return 0, 0
	} else if a > 0 && b > 0 && a > math.MaxInt64/b {
		return 0, +1
	} else if a < math.MinInt64/b {
		return 0, -1
	}
	return a * b, 0
}

func getAngle(x, y float64) float64 {
	return math.Atan2(y, x) * 180 / math.Pi
}

func permutation(n int, k int) int {
	if k > n || k <= 0 {
		panic(fmt.Sprintf("invalid param n:%v k:%v", n, k))
	}
	v := 1
	for i := 0; i < k; i++ {
		v *= (n - i)
	}
	return v
}

/*
	for {

		// Do something

		if !nextPermutation(sort.IntSlice(x)) {
			break
		}
	}
*/
func nextPermutation(x sort.Interface) bool {
	n := x.Len() - 1
	if n < 1 {
		return false
	}
	j := n - 1
	for ; !x.Less(j, j+1); j-- {
		if j == 0 {
			return false
		}
	}
	l := n
	for !x.Less(j, l) {
		l--
	}
	x.Swap(j, l)
	for k, l := j+1, n; k < l; {
		x.Swap(k, l)
		k++
		l--
	}
	return true
}

type combFactorial struct {
	fac    []int
	facinv []int
}

func newcombFactorial(n int) *combFactorial {

	fac := make([]int, n)
	facinv := make([]int, n)
	fac[0] = 1
	facinv[0] = minvfermat(1, mod)

	for i := 1; i < n; i++ {
		fac[i] = mmul(i, fac[i-1])
		facinv[i] = minvfermat(fac[i], mod)
	}

	return &combFactorial{
		fac:    fac,
		facinv: facinv,
	}
}

func (c *combFactorial) factorial(n int) int {
	return c.fac[n]
}

func (c *combFactorial) combination(n, r int) int {
	if r > n {
		return 0
	}
	return mmul(mmul(c.fac[n], c.facinv[r]), c.facinv[n-r])
}

func (c *combFactorial) permutation(n, r int) int {
	if r > n {
		return 0
	}
	return mmul(c.fac[n], c.facinv[n-r])
}

func (c *combFactorial) homogeousProduct(n, r int) int {
	return c.combination(n-1+r, r)
}

func gcd(a, b int) int {
	if b == 0 {
		return a
	}
	return gcd(b, a%b)
}

func divisor(n int) []int {
	sqrtn := int(math.Sqrt(float64(n)))
	c := 2
	divisor := []int{}
	for {
		if n%2 != 0 {
			break
		}
		divisor = append(divisor, 2)
		n /= 2
	}
	c = 3
	for {
		if n%c == 0 {
			divisor = append(divisor, c)
			n /= c
		} else {
			c += 2
			if c > sqrtn {
				break
			}
		}
	}
	if n != 1 {
		divisor = append(divisor, n)
	}
	return divisor
}

type binom struct {
	fac  []int
	finv []int
	inv  []int
}

func newbinom(n int) *binom {
	b := &binom{
		fac:  make([]int, n),
		finv: make([]int, n),
		inv:  make([]int, n),
	}
	b.fac[0] = 1
	b.fac[1] = 1
	b.inv[1] = 1
	b.finv[0] = 1
	b.finv[1] = 1
	for i := 2; i < n; i++ {
		b.fac[i] = b.fac[i-1] * i % mod
		b.inv[i] = mod - mod/i*b.inv[mod%i]%mod
		b.finv[i] = b.finv[i-1] * b.inv[i] % mod
	}
	return b
}

func (b *binom) get(n, r int) int {
	if n < r || n < 0 || r < 0 {
		return 0
	}
	return b.fac[n] * b.finv[r] % mod * b.finv[n-r] % mod
}

// ==================================================
// mod
// ==================================================

func madd(a, b int) int {
	a += b
	if a < 0 {
		a += mod
	} else if a >= mod {
		a -= mod
	}
	return a
}

func mmul(a, b int) int {
	return a * b % mod
}

func mdiv(a, b int) int {
	a %= mod
	return a * minvfermat(b, mod) % mod
}

func mpow(a, n, m int) int {
	if m == 1 {
		return 0
	}
	r := 1
	for n > 0 {
		if n&1 == 1 {
			r = r * a % m
		}
		a, n = a*a%m, n>>1
	}
	return r
}

func minv(a, m int) int {
	p, x, u := m, 1, 0
	for p != 0 {
		t := a / p
		a, p = p, a-t*p
		x, u = u, x-t*u
	}
	x %= m
	if x < 0 {
		x += m
	}
	return x
}

// m only allow prime number
func minvfermat(a, m int) int {
	return mpow(a, m-2, mod)
}

// ==================================================
// binarysearch
// ==================================================

/*
	o = bs(0, len(sl)-1, func(c int) bool {
		return true
	})
*/
func bs(ok, ng int, f func(int) bool) int {
	if !f(ok) {
		return -1
	}
	if f(ng) {
		return ng
	}
	for abs(ok-ng) > 1 {
		mid := (ok + ng) / 2

		if f(mid) {
			ok = mid
		} else {
			ng = mid
		}
	}

	return ok
}

/*
	o = bsfl(0.0, 100.0, 100, func(c float64) bool {
		return true
	})
*/
func bsfl(ok, ng float64, c int, f func(float64) bool) float64 {
	for i := 0; i < c; i++ {

		mid := (ok + ng) / 2

		if f(mid) {
			ok = mid
		} else {
			ng = mid
		}
	}

	return ok
}

// ==================================================
// bit
// ==================================================

func hasbit(a int, n int) bool {
	return (a>>uint(n))&1 == 1
}

func nthbit(a int, n int) int {
	return int((a >> uint(n)) & 1)
}

func popcount(a int) int {
	return bits.OnesCount(uint(a))
}

func bitlen(a int) int {
	return bits.Len(uint(a))
}

func xor(a, b bool) bool { return a != b }

// ==================================================
// string
// ==================================================

func toLowerCase(s string) string {
	return strings.ToLower(s)
}

func toUpperCase(s string) string {
	return strings.ToUpper(s)
}

func isLower(b byte) bool {
	return 'a' <= b && b <= 'z'
}

func isUpper(b byte) bool {
	return 'A' <= b && b <= 'Z'
}

// ==================================================
// sort
// ==================================================

type sortOrder int

const (
	asc sortOrder = iota
	desc
)

func sorti(sl []int) {
	sort.Sort(sort.IntSlice(sl))
}

func sortir(sl []int) {
	sort.Sort(sort.Reverse(sort.IntSlice(sl)))
}

func sorts(sl []string) {
	sort.Slice(sl, func(i, j int) bool {
		return sl[i] < sl[j]
	})
}

type Sort2ArOptions struct {
	keys   []int
	orders []sortOrder
}

type Sort2ArOption func(*Sort2ArOptions)

func opt2ar(key int, order sortOrder) Sort2ArOption {
	return func(args *Sort2ArOptions) {
		args.keys = append(args.keys, key)
		args.orders = append(args.orders, order)
	}
}

// sort2ar(sl,opt2ar(1,asc))
// sort2ar(sl,opt2ar(0,asc),opt2ar(1,asc))
func sort2ar(sl [][2]int, setters ...Sort2ArOption) {
	args := &Sort2ArOptions{}

	for _, setter := range setters {
		setter(args)
	}

	sort.Slice(sl, func(i, j int) bool {
		for idx, key := range args.keys {
			if sl[i][key] == sl[j][key] {
				continue
			}
			switch args.orders[idx] {
			case asc:
				return sl[i][key] < sl[j][key]
			case desc:
				return sl[i][key] > sl[j][key]
			}
		}
		return true
	})
}

// ==================================================
// slice
// ==================================================

func is(l int, def int) []int {
	sl := make([]int, l)
	for i := 0; i < l; i++ {
		sl[i] = def
	}
	return sl
}

//	out(stois("abcde", 'a'))
//	out(stois("abcde", 'a'-1))
//	out(stois("12345", '0'))
func stois(s string, baseRune rune) []int {
	r := make([]int, len(s))
	for i, v := range s {
		r[i] = int(v - baseRune)
	}
	return r
}

func reverse(sl []interface{}) {
	for i, j := 0, len(sl)-1; i < j; i, j = i+1, j-1 {
		sl[i], sl[j] = sl[j], sl[i]
	}
}

func reversei(sl []int) {
	for i, j := 0, len(sl)-1; i < j; i, j = i+1, j-1 {
		sl[i], sl[j] = sl[j], sl[i]
	}
}

func uniquei(sl []int) []int {
	hist := make(map[int]struct{})
	j := 0
	rsl := make([]int, len(sl))
	for i := 0; i < len(sl); i++ {
		if _, ok := hist[sl[i]]; ok {
			continue
		}
		rsl[j] = sl[i]
		hist[sl[i]] = struct{}{}
		j++
	}
	return rsl[:j]
}

// coordinate compression
func cocom(sl []int) ([]int, map[int]int) {
	rsl := uniquei(sl)
	sorti(rsl)
	rm := make(map[int]int)
	for i := 0; i < len(rsl); i++ {
		rm[rsl[i]] = i
	}
	return rsl, rm
}

func popBack(sl []int) (int, []int) {
	return sl[len(sl)-1], sl[:len(sl)-1]
}

func addIdx(pos, v int, sl []int) []int {
	if len(sl) == pos {
		sl = append(sl, v)
		return sl
	}
	sl = append(sl[:pos+1], sl[pos:]...)
	sl[pos] = v
	return sl
}

func delIdx(pos int, sl []int) []int {
	return append(sl[:pos], sl[pos+1:]...)
}

func lowerBound(i int, sl []int) (int, bool) {
	if len(sl) == 0 {
		return 0, false
	}
	idx := bs(0, len(sl)-1, func(c int) bool {
		return sl[c] < i
	})
	if idx == -1 {
		return 0, false
	}
	return idx, true
}

func upperBound(i int, sl []int) (int, bool) {
	if len(sl) == 0 {
		return 0, false
	}
	idx := bs(0, len(sl)-1, func(c int) bool {
		return sl[c] <= i
	})
	if idx == len(sl)-1 {
		return 0, false
	}
	return idx + 1, true
}

// ==================================================
// point
// ==================================================

type point struct {
	x int
	y int
}

type pointf struct {
	x float64
	y float64
}

func (p point) isValid(x, y int) bool {
	return 0 <= p.x && p.x < x && 0 <= p.y && p.y < y
}

func pointAdd(a, b point) point {
	return point{a.x + b.x, a.y + b.y}
}

func pointDist(a, b point) float64 {
	return math.Sqrt(float64((a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y)))
}

func pointfDist(a, b pointf) float64 {
	return math.Sqrt((a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y))
}

// ==================================================
// queue
// ==================================================

/*
	q := list.New()
	q.PushBack(val)
	e := q.Front()
	for e != nil {
		t := e.Value.(int)

		// Do something

		e = e.Next()
    }
*/

// ==================================================
// heap
// ==================================================

/*
	h := &IntHeap{}
	heap.Init(h)
	heap.Push(h, 2)
	heap.Push(h, 3)
	out(heap.Pop(h).(int))
	out(h.Min().(int))
	heap.Pop(h)
	out(h.Len())
*/
type IntHeap []int

func (h IntHeap) Len() int           { return len(h) }
func (h IntHeap) Less(i, j int) bool { return h[i] < h[j] }
func (h IntHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *IntHeap) Push(x interface{}) {
	*h = append(*h, x.(int))
}

func (h *IntHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

func (h *IntHeap) IsEmpty() bool {
	return h.Len() == 0
}

func (h *IntHeap) Min() interface{} {
	return (*h)[0]
}

type pq struct {
	arr   []interface{}
	comps []compFunc
}

/*
	pq := newpq([]compFunc{func(p, q interface{}) int {
		if p.(edge).cost < q.(edge).cost {
			return -1
		} else if p.(edge).cost == q.(edge).cost {
			return 0
		}
		return 1
	}})
	heap.Init(pq)
	heap.Push(pq, edge{from: 3, to: 3, cost: 2})
	heap.Push(pq, edge{from: 2, to: 2, cost: 3})
	out(heap.Pop(pq).(edge))
	out(pq.Min().(edge))
	heap.Pop(pq)
	out(pq.Len())
*/
type compFunc func(p, q interface{}) int

func newpq(comps []compFunc) *pq {
	return &pq{
		comps: comps,
	}
}

func (pq pq) Len() int {
	return len(pq.arr)
}

func (pq pq) Swap(i, j int) {
	pq.arr[i], pq.arr[j] = pq.arr[j], pq.arr[i]
}

func (pq pq) Less(i, j int) bool {
	for _, comp := range pq.comps {
		result := comp(pq.arr[i], pq.arr[j])
		switch result {
		case -1:
			return true
		case 1:
			return false
		case 0:
			continue
		}
	}
	return true
}

func (pq *pq) Push(x interface{}) {
	pq.arr = append(pq.arr, x)
}

func (pq *pq) Pop() interface{} {
	n := pq.Len()
	item := pq.arr[n-1]
	pq.arr = pq.arr[:n-1]
	return item
}

func (pq *pq) IsEmpty() bool {
	return pq.Len() == 0
}

func (pq *pq) Min() interface{} {
	return pq.arr[0]
}

// ==================================================
// cusum
// ==================================================

type cusum struct {
	s []int
}

func newcusum(sl []int) *cusum {
	c := &cusum{}
	c.s = make([]int, len(sl)+1)
	for i, v := range sl {
		c.s[i+1] = c.s[i] + v
	}
	return c
}

func (c *cusum) get(f, t int) int {
	return c.s[t+1] - c.s[f]
}

/*
	mp := make([][]int, n)
	for i := 0; i < k; i++ {
		mp[i] = make([]int, m)
	}
	cusum2d := newcusum2d(sl)
	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			t:=cusum2d.get(0, 0, i, j)
		}
	}
*/

type cusum2d struct {
	s [][]int
}

func newcusum2d(sl [][]int) *cusum2d {
	c := &cusum2d{}
	n := len(sl)
	m := len(sl[0])
	c.s = make([][]int, n+1)
	for i := 0; i < n+1; i++ {
		c.s[i] = make([]int, m+1)
	}
	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			c.s[i+1][j+1] = c.s[i+1][j] + c.s[i][j+1] - c.s[i][j]
			c.s[i+1][j+1] += sl[i][j]
		}
	}
	return c
}

// x1 <= x <= x2, y1 <= y <= y2
func (c *cusum2d) get(x1, y1, x2, y2 int) int {
	return c.s[x2][y2] + c.s[x1][y1] - c.s[x1][y2] - c.s[x2][y1]
}

// ==================================================
// union find
// ==================================================

type unionFind struct {
	par []int
}

func newUnionFind(n int) *unionFind {
	u := &unionFind{
		par: make([]int, n),
	}
	for i := range u.par {
		u.par[i] = -1
	}
	return u
}

func (u *unionFind) root(x int) int {
	if u.par[x] < 0 {
		return x
	}
	u.par[x] = u.root(u.par[x])
	return u.par[x]
}

func (u *unionFind) unite(x, y int) {
	x = u.root(x)
	y = u.root(y)
	if x == y {
		return
	}
	if u.size(x) < u.size(y) {
		x, y = y, x
	}
	u.par[x] += u.par[y]
	u.par[y] = x
}

func (u *unionFind) issame(x, y int) bool {
	if u.root(x) == u.root(y) {
		return true
	}
	return false
}

func (u *unionFind) size(x int) int {
	return -u.par[u.root(x)]
}

// ==================================================
// bit
// ==================================================

type bit struct {
	n int
	b []int
}

func newbit(n int) *bit {
	return &bit{
		n: n + 1,
		b: make([]int, n+1),
	}
}

func (b *bit) add(i, x int) {
	for i++; i < b.n && i > 0; i += i & -i {
		b.b[i] += x
	}
}

func (b *bit) sum(i int) int {
	ret := 0
	for i++; i > 0; i -= i & -i {
		ret += b.b[i]
	}
	return ret
}

// l <= x < r
func (b *bit) rangesum(l, r int) int {
	return b.sum(r-1) - b.sum(l-1)
}

func (b *bit) lowerBound(x int) int {
	idx, k := 0, 1
	for k < b.n {
		k <<= 1
	}
	for k >>= 1; k > 0; k >>= 1 {
		if idx+k < b.n && b.b[idx+k] < x {
			x -= b.b[idx+k]
			idx += k
		}
	}
	return idx
}

// ==================================================
// segment tree
// ==================================================

type streeculctype int

var stadd streeculctype = 1
var stset streeculctype = 2

type streeminmmax int

var stmin streeminmmax = 1
var stmax streeminmmax = 2

/*
s := newstree(n,stmin|stmax,stset|stadd)
s.set(i,x)
s.add(i,x)
result1 := s.query(l,r)
result2 := s.findrightest(l,r,x)
result3 := s.findlefttest(l,r,x)
*/
type stree struct {
	n    int
	b    []int
	def  int
	cmp  func(i, j int) int
	culc func(i, j int) int
}

func newstree(n int, minmax streeminmmax, ctype streeculctype) *stree {
	tn := 1
	for tn < n {
		tn *= 2
	}
	s := &stree{
		n: tn,
		b: make([]int, 2*tn-1),
	}
	switch minmax {
	case stmin:
		s.def = inf
		for i := 0; i < 2*tn-1; i++ {
			s.b[i] = s.def
		}
		s.cmp = func(i, j int) int {
			return min(i, j)
		}
	case stmax:
		s.cmp = func(i, j int) int {
			return max(i, j)
		}
	}
	switch ctype {
	case stadd:
		s.culc = func(i, j int) int {
			return i + j
		}
	case stset:
		s.culc = func(i, j int) int {
			return j
		}
	}
	return s
}

func (s *stree) add(i, x int) {
	i += s.n - 1
	s.b[i] += x

	for i > 0 {
		i = (i - 1) / 2
		s.b[i] = s.cmp(s.b[i*2+1], s.b[i*2+2])
	}
}

func (s *stree) set(i, x int) {
	i += s.n - 1
	s.b[i] = x

	for i > 0 {
		i = (i - 1) / 2
		s.b[i] = s.cmp(s.b[i*2+1], s.b[i*2+2])
	}
}

func (s *stree) query(a, b int) int {
	return s.querysub(a, b, 0, 0, s.n)
}

func (s *stree) querysub(a, b, k, l, r int) int {
	if r <= a || b <= l {
		return s.def
	}
	if a <= l && r <= b {
		return s.b[k]
	}
	return s.cmp(
		s.querysub(a, b, k*2+1, l, (l+r)/2),
		s.querysub(a, b, k*2+2, (l+r)/2, r),
	)
}

func (s *stree) findrightest(a, b, x int) int {
	return s.findrightestsub(a, b, x, 0, 0, s.n)
}

func (s *stree) findrightestsub(a, b, x, k, l, r int) int {
	if s.b[k] > x || r <= a || b <= l {
		return a - 1
	} else if k >= s.n-1 {
		return k - s.n + 1
	}
	vr := s.findrightestsub(a, b, x, 2*k+2, (l+r)/2, r)
	if vr != a-1 {
		return vr
	}
	return s.findrightestsub(a, b, x, 2*k+1, l, (l+r)/2)
}

func (s *stree) findleftest(a, b, x int) int {
	return s.findleftestsub(a, b, x, 0, 0, s.n)
}

func (s *stree) findleftestsub(a, b, x, k, l, r int) int {
	if s.b[k] > x || r <= a || b <= l {
		return b
	} else if k >= s.n-1 {
		return k - s.n + 1
	}
	vl := s.findleftestsub(a, b, x, 2*k+1, l, (l+r)/2)
	if vl != b {
		return vl
	}
	return s.findleftestsub(a, b, x, 2*k+2, (l+r)/2, r)
}

func (s *stree) debug() {
	l := []string{}
	t := 2
	out("data")
	for i := 0; i < 2*s.n-1; i++ {
		if i+1 == t {
			t *= 2
			out(strings.Join(l, " "))
			l = []string{}
		}
		if s.b[i] == inf {
			l = append(l, "∞")
		} else {
			l = append(l, strconv.Itoa(s.b[i]))
		}
	}
	out(strings.Join(l, " "))
}

/*
s := newlazystree(n,stmin|stmax,stset|stadd)
s.set(i,x)
s.add(i,x)
s.rc(l,r,x)
result1 := s.query(l,r)
result2 := s.findrightest(l,r,x)
result3 := s.findlefttest(l,r,x)
*/
type lazystree struct {
	on   int
	n    int
	b    []int
	lazy []int
	def  int
	cmp  func(i, j int) int
	culc func(i, j int) int
}

func newlazystree(n int, minmax streeminmmax, ctype streeculctype) lazystree {
	tn := 1
	for tn < n {
		tn *= 2
	}
	s := lazystree{
		on:   n,
		n:    tn,
		b:    make([]int, 2*tn-1),
		lazy: make([]int, 2*tn-1),
	}
	switch minmax {
	case stmin:
		s.def = inf
		for i := 0; i < 2*tn-1; i++ {
			s.b[i] = s.def
			s.lazy[i] = s.def
		}
		s.cmp = func(i, j int) int {
			return min(i, j)
		}
	case stmax:
		s.cmp = func(i, j int) int {
			return max(i, j)
		}
	}
	switch ctype {
	case stadd:
		s.culc = func(i, j int) int {
			if i == s.def {
				return j
			}
			if i == s.def {
				return i
			}
			return i + j
		}
	case stset:
		s.culc = func(i, j int) int {
			return j
		}
	}
	return s
}

func (s lazystree) eval(k int) {
	if s.lazy[k] == s.def {
		return
	}
	if k < s.n-1 {
		s.lazy[k*2+1] = s.culc(s.lazy[k*2+1], s.lazy[k])
		s.lazy[k*2+2] = s.culc(s.lazy[k*2+2], s.lazy[k])
	}
	s.b[k] = s.culc(s.b[k], s.lazy[k])
	s.lazy[k] = s.def
}

func (s lazystree) add(i, x int) {
	i += s.n - 1
	s.b[i] += x

	for i > 0 {
		i = (i - 1) / 2
		s.b[i] = s.cmp(s.b[i*2+1], s.b[i*2+2])
	}
}

func (s lazystree) set(i, x int) {
	i += s.n - 1
	s.b[i] = x

	for i > 0 {
		i = (i - 1) / 2
		s.b[i] = s.cmp(s.b[i*2+1], s.b[i*2+2])
	}
}

// range culc
func (s lazystree) rc(a, b, x int) {
	s.rcsub(a, b, x, 0, 0, s.n)
}

func (s lazystree) rcsub(a, b, x, k, l, r int) {
	s.eval(k)
	if a <= l && r <= b {
		s.lazy[k] = s.culc(s.lazy[k], x)
		s.eval(k)
	} else if l < b && a < r {
		s.rcsub(a, b, x, k*2+1, l, (l+r)/2)
		s.rcsub(a, b, x, k*2+2, (l+r)/2, r)
		s.b[k] = s.cmp(s.b[k*2+1], s.b[k*2+2])
	}
}

func (s lazystree) get(a int) int {
	return s.query(a, a+1)
}

func (s lazystree) query(a, b int) int {
	return s.querysub(a, b, 0, 0, s.n)
}

func (s lazystree) querysub(a, b, k, l, r int) int {
	s.eval(k)
	if r <= a || b <= l {
		return s.def
	}
	if a <= l && r <= b {
		return s.b[k]
	}
	return s.cmp(
		s.querysub(a, b, k*2+1, l, (l+r)/2),
		s.querysub(a, b, k*2+2, (l+r)/2, r),
	)
}

func (s lazystree) findrightest(a, b, x int) int {
	return s.findrightestsub(a, b, x, 0, 0, s.n)
}

func (s lazystree) findrightestsub(a, b, x, k, l, r int) int {
	if s.b[k] > x || r <= a || b <= l {
		return a - 1
	} else if k >= s.n-1 {
		return k - s.n + 1
	}
	vr := s.findrightestsub(a, b, x, 2*k+2, (l+r)/2, r)
	if vr != a-1 {
		return vr
	}
	return s.findrightestsub(a, b, x, 2*k+1, l, (l+r)/2)
}

func (s lazystree) findleftest(a, b, x int) int {
	return s.findleftestsub(a, b, x, 0, 0, s.n)
}

func (s lazystree) findleftestsub(a, b, x, k, l, r int) int {
	if s.b[k] > x || r <= a || b <= l {
		return b
	} else if k >= s.n-1 {
		return k - s.n + 1
	}
	vl := s.findleftestsub(a, b, x, 2*k+1, l, (l+r)/2)
	if vl != b {
		return vl
	}
	return s.findleftestsub(a, b, x, 2*k+2, (l+r)/2, r)
}

func (s lazystree) debug() {
	l := []string{}
	t := 2
	out("data")
	for i := 0; i < 2*s.n-1; i++ {
		if i+1 == t {
			t *= 2
			out(strings.Join(l, " "))
			l = []string{}
		}
		if s.b[i] == inf {
			l = append(l, "∞")
		} else {
			l = append(l, strconv.Itoa(s.b[i]))
		}
	}
	out(strings.Join(l, " "))
	out("lazy")
	l = []string{}
	t = 2
	for i := 0; i < 2*s.n-1; i++ {
		if i+1 == t {
			t *= 2
			out(strings.Join(l, " "))
			l = []string{}
		}
		if s.lazy[i] == inf {
			l = append(l, "∞")
		} else {
			l = append(l, strconv.Itoa(s.lazy[i]))
		}
	}
	out(strings.Join(l, " "))
}

func (s lazystree) debug2() {
	l := make([]string, s.n)
	for i := 0; i < s.on; i++ {
		l[i] = strconv.Itoa(s.get(i))
	}
	out(strings.Join(l, " "))
}

// ==================================================
// tree
// ==================================================

type tree struct {
	size       int
	root       int
	edges      [][]edge
	parentsize int
	parent     [][]int
	depth      []int
	orderidx   int
	order      []int
}

/*
	n := ni()
	edges := make([][]edge, n)
	for i := 0; i < n-1; i++ {
		s, t := ni2()
		s--
		t--
		edges[s] = append(edges[s], edge{to: t})
		edges[t] = append(edges[t], edge{to: s})
	}
	tree := newtree(n, 0, edges)
	tree.init()
*/
func newtree(size int, root int, edges [][]edge) *tree {
	parentsize := int(math.Log2(float64(size))) + 1
	parent := make([][]int, parentsize)
	for i := 0; i < parentsize; i++ {
		parent[i] = make([]int, size)
	}
	depth := make([]int, size)
	order := make([]int, size)
	return &tree{
		size:       size,
		root:       root,
		edges:      edges,
		parentsize: parentsize,
		parent:     parent,
		depth:      depth,
		order:      order,
	}
}

func (t *tree) init() {
	t.dfs(t.root, -1, 0)
	for i := 0; i+1 < t.parentsize; i++ {
		for j := 0; j < t.size; j++ {
			if t.parent[i][j] < 0 {
				t.parent[i+1][j] = -1
			} else {
				t.parent[i+1][j] = t.parent[i][t.parent[i][j]]
			}
		}
	}
}

func (t *tree) dfs(v, p, d int) {
	t.order[v] = t.orderidx
	t.orderidx++
	t.parent[0][v] = p
	t.depth[v] = d
	for _, nv := range t.edges[v] {
		if nv.to != p {
			t.dfs(nv.to, v, d+1)
		}
	}
}

func (t *tree) auxiliarytree(sl []int) []int {
	sort.Slice(sl, func(i, j int) bool { return t.order[sl[i]] < t.order[sl[j]] })
	return sl
}

// ==================================================
// graph
// ==================================================

type edge struct {
	from int
	to   int
	cost int
	rev  int
}
