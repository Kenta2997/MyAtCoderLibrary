// https://yukicoder.me/submissions/693392 
// https://judge.u-aizu.ac.jp/onlinejudge/review.jsp?rid=5829941#1    LCAあり hl.HLvector[hl.index_LCA]
// https://atcoder.jp/contests/abc133/submissions/25368840
struct HL{ // 0indexed
	vector<int> HLvector; // 連結成分毎に並べた時,左からi+1番目にある頂点
	vector<int> root_union; // root_union[i] = HLした後 頂点 HLvector[i] の根の頂点番号
	vector<int> par; // par[i]=頂点iの親 HL分解する前のグラフで1つ根に向かったやつ
	vector<int> index; // index[i]=j <=> HLvector[j]=i (頂点iが何番目にあるか)
	vector<int> depth; //depth[i]=頂点iの元の木での深さ
	vector<int> num; // rootがiの部分木のサイズ
	int index_LCA;//
	void make_HLvector(const vector<int> &u,const vector<int> &v){
		HLvector.clear();
		root_union.clear();
		int N = u.size() + 1;
		par.resize(N);
		for(int i=0;i<N;i++) par[i] = -1;
		auto ltof = leaf_to_root(u,v,0);
		num.resize(N);
		for(int i=0;i<N;i++) num[i] = 1;
		vector<vector<int>> G(N);
		for(auto e:ltof) num[e.first] += num[e.second];
		for(auto e:ltof) par[e.second] = e.first;
		for(int i=0;i<N-1;i++){
			G[u[i]].push_back(v[i]);
			G[v[i]].push_back(u[i]);
		}
		stack< pair<int,int> > st;
		st.push({0,0}); // 頂点,最も浅い頂点
		while(st.size()){
			int i = st.top().first;
			int p = st.top().second;
			st.pop();
			HLvector.push_back(i);
			root_union.push_back(p);
			int val = 0 , big_child = 0;
			for(auto v:G[i]){
				if(num[v] > num[i]) continue; // 親には戻らない
				if(num[v] > val){
					val = num[v];
					big_child = v;
				}
			}
			if(val == 0) continue;
			for(auto v:G[i]){
				if(num[v] > num[i] || v == big_child) continue;
				st.push({v,v});
			}
			st.push({big_child,p});
		}
		index.resize(N);
		for(int i=0;i<N;i++){
			index[ HLvector[i] ] = i;
		}
		reverse(ltof.begin(), ltof.end());
		depth.resize(N);
		depth[0] = 0;
		for(auto e:ltof) depth[e.second] = depth[e.first] + 1;
	}
 
	vector< pair<int,int> > index_route(int u,int v){
		// u,vパスを連結成分で区切った時のパスの(HLvectorの)indexを示す
		vector< pair<int,int> > res;
		while(root_union[index[u]] != root_union[index[v]]){
			if(depth[ root_union[index[u]] ] < depth[ root_union[index[v]] ]) swap(u,v);
			int a = root_union[ index[u] ];
			//u と a のindex
			res.push_back({  index[a]  , index[u]  });
			u = par[a];
			assert(u>=0);
		}
		res.push_back({min(index[u],index[v]),max(index[u],index[v])});
		index_LCA = index[u];
		if(depth[ u ] > depth[ v ]) index_LCA = index[v];
		return res;
	}
	int LCA(int u,int v){
		index_route(u,v);
		return HLvector[index_LCA];
	}
	pair<int,int> subtree_index(int i){ // HLvectorで頂点iの部分木のインデックスの両端 未varify
		return {index[i],index[i]+num[i]}; // [res.first,res.second-1]
	}
 
	vector< pair<int,int> > leaf_to_root(const vector<int> &u,const vector<int> &v,const int root){
		// return <parent,child>
		int yet = -1;
		vector<int> d(u.size()+1,yet);
		vector< pair<int,int> > res;
		queue<int> Q;
		Q.push(root);
		d[root] = 0;
		vector< vector<int> > G(u.size()+1);
		for(int i=0;i<u.size();i++){
			G[u[i]].push_back(v[i]);
			G[v[i]].push_back(u[i]);
		}
		while( Q.size() ){
			int now = Q.front();
			Q.pop();
			for(auto nex:G[now]){
				if(d[nex]!=yet)continue;
				d[nex] = d[now] + 1;
				Q.push(nex);
				res.push_back( make_pair(now,nex) );
			}
		}
		reverse(res.begin(), res.end());
		return res;// return <parent,child>
	}
};
