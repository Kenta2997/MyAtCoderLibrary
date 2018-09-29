class UnionFind{
public:
	vector<int> par;
	//0-indexed
	void init(int n = 1) {
		par.resize(n);
		for (int i = 0; i < n; ++i) par[i] = i;
	}
	int root(int x = 1){
		if(par[x]==x)return x;
		else{
			return par[x] = root(par[x]);
		}
	}
	void unite(int x = 1,int y = 1){
		x = root(x);
		y = root(y);
		if(x==y)return;
		else par[x] = y;
	}
};