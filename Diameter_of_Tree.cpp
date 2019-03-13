// http://judge.u-aizu.ac.jp/onlinejudge/review.jsp?rid=3424270
// http://techtipshoge.blogspot.com/2016/08/blog-post.html
struct Diameter_of_Tree{
	int V; // 0-indexd
	vector< vector< pair<int,int> > > g;
	Diameter_of_Tree(int V_){
		V = V_;
		g.resize(V);
	}
	void add_edge(int v1,int v2,int cost){
		g[v1].push_back({v2,cost});
		g[v2].push_back({v1,cost});
	}
	pair<int,int> tddfs(int v,int par = -1){
		pair<int,int> ret = {0,v};
		for(auto &x : g[v]){
			int w,cost;
			tie(w,cost) = x;
			if( w == par )continue;
			auto r = tddfs(w,v);
			ret = max(ret,{cost+r.first,r.second});
		}
		return ret;
	}
	int treeDiameter(){
		auto v = tddfs(0);
		auto w = tddfs(v.second);
		return w.first;
	}
};