// http://judge.u-aizu.ac.jp/onlinejudge/review.jsp?rid=3178606#1
// ant p.286
struct SCC{
	int V;
	vector< vector<int> > G;
	vector< vector<int> > rG;
	vector<int> vs;
	vector<bool> used;
	vector<int> cmp;
	SCC(int v){
		V = v;
		G.resize(V);
		rG.resize(V);
		used.resize(V,false);
		cmp.resize(V);
	}

	void add_edge(int from,int to){
		G[from].push_back(to);
		rG[to].push_back(from);
	}
	void dfs(int v){
		used[v] = true;
		for(int i=0;i<G[v].size();i++){
			if(!used[G[v][i]])dfs(G[v][i]);
		}
		vs.push_back(v);
	}
	void rdfs(int v,int k){
		used[v] = true;
		cmp[v] = k;
		for(int i = 0; i < rG[v].size(); ++i){
			if(!used[rG[v][i]])rdfs(rG[v][i],k);
		}
	}
	int scc(){
		used = vector<bool>(V,false);
		vs.clear();
		for (int v = 0; v < V; ++v){
			if(!used[v])dfs(v);
		}
		used = vector<bool>(V,false);
		int k = 0;
		for (int i = vs.size() - 1; i >= 0 ; --i){
			if(!used[vs[i]])rdfs(vs[i],k++);
		}
		return k;
	}
	bool is_same(int v1,int v2){
		return cmp[v1]==cmp[v2];
	}
};