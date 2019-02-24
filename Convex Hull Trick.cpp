// http://kazuma8128.hatenablog.com/entry/2018/02/28/102130
// https://atcoder.jp/contests/dp/submissions/4362019
template <typename T, const T id>
class CHT {
	struct line {
		T a, b;
		line(T a_ = 0, T b_ = 0) : a(a_), b(b_) {}
		T get(T x) { return a * x + b; }
	};
	struct node {
		line l;
		node *lch, *rch;
		node(line l_) : l(l_), lch(nullptr), rch(nullptr) {}
		~node() {
			if (lch) delete lch;
			if (rch) delete rch;
		}
	};
 
private:
	const int n;
	const vector<T> pos;
	node *root;
 
public:
	CHT(const vector<T>& pos_) : n(pos_.size()), pos(pos_), root(nullptr) {}
	~CHT() {
		if (root) delete root;
	}
	void insert(T a, T b) {
		line l(a, b);
		root = modify(root, 0, n, l);
	}
	T get(T x) const {
		int t = lower_bound(pos.begin(), pos.end(), x) - pos.begin();
		assert(t < n && pos[t] == x);
		return sub(root, 0, n, t);
	}
 
private:
	node* modify(node *p, int lb, int ub, line& l) {
		if (!p) return new node(l);
		if (p->l.get(pos[lb]) >= l.get(pos[lb]) && p->l.get(pos[ub - 1]) >= l.get(pos[ub - 1])) return p;
		if (p->l.get(pos[lb]) <= l.get(pos[lb]) && p->l.get(pos[ub - 1]) <= l.get(pos[ub - 1])) {
			p->l = l;
			return p;
		}
		int c = (lb + ub) / 2;
		if (p->l.get(pos[c]) < l.get(pos[c])) swap(p->l, l);
		if (p->l.get(pos[lb]) <= l.get(pos[lb]))
			p->lch = modify(p->lch, lb, c, l);
		else
			p->rch = modify(p->rch, c, ub, l);
		return p;
	}
	T sub(node *p, int lb, int ub, int t) const {
		if (!p) return id;
		if (ub - lb == 1) return p->l.get(pos[t]);
		int c = (lb + ub) / 2;
		if (t < c) return max(p->l.get(pos[t]), sub(p->lch, lb, c, t));
		return max(p->l.get(pos[t]), sub(p->rch, c, ub, t));
	}
};