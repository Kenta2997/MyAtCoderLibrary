// How to use
// http://judge.u-aizu.ac.jp/onlinejudge/review.jsp?rid=3095512#1
// https://www.slideshare.net/hcpc_hokudai/binary-indexed-tree

template <typename T>
struct twodimBIT
{
// 1-indexed
private:
	vector< vector<T> > array;
	const int n;
	const int m;

public:
	twodimBIT(int _n,int _m):
		array(_n+1,vector<T>(_m+1,0)), n(_n),m(_m){}

	// (1,1) ~ (x,y) sum
	T sum(int x,int y) {
		T s = 0;
		for(int i=x;i>0;i-=i&(-i))
			for(int j=y;j>0;j-=j&(-j))
				s += array[i][j];
		return s;
	}

	// [(x1,y1),(x2,y2)] -> (x1<x2,y1<y2)
	T sum(int x1,int y1,int x2,int y2){
		return sum(x2,y2) - sum(x1-1,y2) - sum(x2,y1-1) + sum(x1-1,y1-1);
	}

	void add(int x,int y,T k) {
		for(int i=x;i<=n;i+=i&(-i))
			for(int j=y;j<=m;j+=j&(-j))
				array[i][j] += k;
	}
};