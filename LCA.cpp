
// https://beta.atcoder.jp/contests/abc014/submissions/3369070
// This Seg_Tree class returns min value and index.
// ant book p.294

class Seg_Tree{
public:
	vector< pair<int,int> > dat;
	// first  -> min value
	// second -> min index
	int n;
	void initsize(int n0){
		int k=1;
		while(1){
			if(n0<=k){
				n=k;
				dat.resize(2*n-1);
				for(int i=0;i<2*n-1;i++){
					dat[i].first=INT_MAX;
					dat[i].second = i-n+1;
				}
				break;
			}
			k*=2;
		}
	}
	//i banme wo x nisuru
	void update(int i,int x){
		i += n-1;
		dat[i].first = x;
		while(i>0){
			i = (i-1) / 2;
			if(dat[i*2+1].first<=dat[i*2+2].first)
				dat[i] = dat[i*2+1];
			else
				dat[i] = dat[i*2+2];
		}
	}
	//[a,b)
	pair<int,int> query0(int a,int b,int k,int l,int r){
		if(r<=a || b<=l)return {INT_MAX,-1};
		if(a<=l && r<=b)return dat[k];
		else{
			pair<int,int> vl = query0(a,b,k*2+1,l,(l+r)/2);
			pair<int,int> vr = query0(a,b,k*2+2,(l+r)/2,r);
			if(vl.first<=vr.first)return vl;
			else return vr;
		}
	}
	//return min [a,b)
	pair<int,int> query(int a,int b){
		return query0(a,b,0,0,n);
	}
};
 
 
 
struct LCA{
	// This struct uses class Seg_Tree
	vector< vector<ll> > G;
	ll root;
	vector<ll> vs,depth,id;
	Seg_Tree se;
 
	LCA(ll v){
		G.resize(v);
		vs.resize(v*2-1);
		depth.resize(v*2-1);
		id.resize(v);
		root = 0;
	}
 
	void dfs(ll v,ll p,ll d,ll &k){
		id[v] = k;
		vs[k] = v;
		depth[k++] = d;
		for(ll i=0;i<G[v].size();i++){
			if(G[v][i] != p){
				dfs(G[v][i],v,d+1,k);
				vs[k] = v;
				depth[k++] = d;
			}
		}
	}
 
	void init(){
		ll V = id.size();
		ll k = 0;
		dfs(root,-1,0,k);
		se.initsize(V*2-1);
		for(int i=0;i<depth.size();i++){
			se.update(i,depth[i]);
		}
	}

	int lca(ll u,ll v){
		return vs[se.query(min(id[u],id[v]),max(id[u],id[v])+1).second];
	}
 
	int length_from_root(int v){
		return depth[id[v]];
	}
};