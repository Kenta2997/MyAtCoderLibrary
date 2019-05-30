// https://camypaper.bitbucket.io/2016/12/18/adc2016/
// https://www65.atwiki.jp/kyopro-lib/pages/13.html
// find:O(logN) unite:O(logN)
// t回目までのuniteクエリでx,yが同じ連結成分にいるかどうかを求められる

// https://atcoder.jp/contests/code-thanks-festival-2017-open/submissions/5680215


class persistentUF {
public:
	const static int MAX_N = 100010;
	unordered_map<int, int> par[MAX_N];
	int rank[MAX_N];
	int fin[MAX_N];
	int idx;
	persistentUF() { init(); }
	void init() {
		idx = 0;
		for(int i=0;i<MAX_N;++i) par[i][0] = i, rank[i] = 1, fin[i] = 0;
	}
	persistentUF(int n) { init(n); }
	void init(int n) {
		idx = 0;
		for(int i=0;i<n;++i) par[i][0] = i, rank[i] = 1, fin[i] = 0;
	}
	int find(int x, int t) {
		if(t >= fin[x] && par[x][fin[x]] != x) return find(par[x][fin[x]], t);
		return x;
	}
	void unite(int x, int y) {
		x = find(x, idx);
		y = find(y, idx);
		idx++;
		if(x == y) return;
		if(rank[x] < rank[y]) par[x][idx] = y, fin[x] = idx;
		else {
			par[y][idx] = x, fin[y] = idx;
			if(rank[x] == rank[y]) rank[x]++;
		}
	}
	bool same(int x, int y, int t) { return find(x, t) == find(y, t); }
};