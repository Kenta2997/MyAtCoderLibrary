//1 UnionFind
//2 Seg_Tree  //and index abc014D
//3 dijkstra
//4 topolodical
//5 combination fast?
//6 Maxflow
//7 MinimumCostflow
//8 Inversions - tentosu
//9 BIT
//10 kosokufurie
//11 nap same ok
//12 twodimBIT
//13 Knap With Limitations
//14 Bellman-Ford
//15 強連結成分分解
//16 CHT
//17 CHT
//18 無向グラフの橋と関節点O(E+V),列挙するためにsortしてるからO(ElonE+VlogV)になってる,個数だけの時はcalc()内のsort消してください
//19 木の直径
//20 座標圧縮
//21 RollingHash
//22 RSQ
//23 平方分割RUQ  forはintを使うと早い,cin,cout高速化も重要 (https://atcoder.jp/contests/abl/submissions/25658689) 
//24 // 区間加算　一点取得 RAQ 平方分割で実装
//25 //遅延セグ木
//26 //ダブリング urlのみ https://atcoder.jp/contests/abc167/submissions/13130493
//27 nCr 任意mod->Lucasの定理 , 別でpが素数の時は pの乗数だけ別で数える https://atcoder.jp/contests/arc117/submissions/21881563 
//28 2D imos https://atcoder.jp/contests/typical90/submissions/23900982 
//29 最近点対 https://onlinejudge.u-aizu.ac.jp/status/users/fullhouse1987/submissions/1/CGL_5_A/judge/5789282/C++14
//30 HL分解(LCAもある) 1つのパスクエリを直線クエリO(logN)個に分解 https://yukicoder.me/submissions/693392
//31 ガウスジョルダン　掃き出し法( https://drken1215.hatenablog.com/entry/2019/03/20/202800 ) ACmod2自由度のみ  https://atcoder.jp/contests/typical90/submissions/25389864   https://atcoder.jp/contests/code-thanks-festival-2017/submissions/25390046
//32 形式的べき級数　https://judge.yosupo.jp/submission/42413 optさんのやつ　使い慣れてない
  ///    .使い方参考1  https://judge.yosupo.jp/problem/pow_of_formal_power_series
  ///    .使い方参考2  https://opt-cp.com/fps-implementation/
  ///    .ALC参考     https://github.com/TumoiYorozu/single-file-ac-library
  ///    .10^9+7提出  https://atcoder.jp/contests/abc178/submissions/25741104
  ///    .998244353提出　 https://atcoder.jp/contests/abc179/submissions/25740540
  ///    .998244353提出   https://atcoder.jp/contests/kupc2020/submissions/25793725  二項係数も使ってるカタラン数
//33

///////111//////////////////////////////////////////////////////////
// 型なんでもで必要なものだけ https://atcoder.jp/contests/abc214/submissions/25088564
class UnionFind{
	vector<int> par;
	//0-indexed
	void init(int n){
		component = n;
		par.resize(n,-1);
		msiz = 1;
	}
public:
	int msiz,component;
	UnionFind(int n_){init(n_);}
	int root(int x){
		if(par[x]<0)return x;
		return par[x] = root(par[x]);
	}
	void unite(int x,int y){
		x = root(x);
		y = root(y);
		if(x==y)return;
		if(size(x) < size(y)) swap(x,y);
		// x <- y
		par[x] += par[y];
		par[y] = x;
		msiz = max(msiz,-par[x]);
		component--;
	}
	int size(int x){
		return -par[root(x)];
	}
	bool same(int x,int y){
		return root(x)==root(y);
	}
};

//////////111//////////////222//////////////////////////////////////
// http://judge.u-aizu.ac.jp/onlinejudge/review.jsp?rid=2871291
// https://atcoder.jp/contests/arc017/submissions/10729573
template <typename T>
class Seg_Tree{
public: // 0-index
	vector<T> dat;
	T initial,M;
	int n;
	T unite(T a,T b){//
		return a + b;
	}
	Seg_Tree(int n0_=1,T initial_=1,T M_=1){
		initsize(n0_,initial_,M_);
	}
	void initsize(int n0,T initial_,T M_){
		M = M_;
		initial = initial_;
		int k=1;
		while(1){
			if(n0<=k){
				n=k;
				dat.resize(2*n-1);
				for(int i=0;i<2*n-1;i++)dat[i]=initial;
				break;
			}
			k*=2;
		}
	}
 
	//i banme wo x nisuru
	void update(int i,T x){
		i += n-1;
		dat[i] = x;
		while(i>0){
			i = (i-1) / 2;
			dat[i] = unite(dat[i*2+1],dat[i*2+2]);
		}
	}
 
	//[a,b)
	T query0(int a,int b,int k,int l,int r){
		if(r<=a || b<=l)return initial;
		if(a<=l && r<=b)return dat[k];
		else{
			T vl = query0(a,b,k*2+1,l,(l+r)/2);
			T vr = query0(a,b,k*2+2,(l+r)/2,r);
			return unite(vl,vr);
		}
	}
 
	//return [a,b)
	T query(int a,int b){
		return query0(a,b,0,0,n);
	}
};
//////////////222////////////////333///////////////////////////////
class dijkstra{ 
//0indexed 
public: 
	int v,startv; 
	vector<int> d; 
	vector<vector<pair<int,int> > >e; 

	void initsize(int n0){ 
		d.resize(n0); 
		for(int i=0;i<n0;i++)d[i]=INT_MAX;
		vector<pair<int,int> > ep; 
		for(int i=0;i<n0;i++)e.push_back(ep); 
		v=n0; 
	} 

	void initstart(int s){ 
		startv=s;
		d[s]=0;
	}

	void make_edge(int x,int y,int cost){ 
		e[x].push_back(make_pair(y,cost)); 
		e[y].push_back(make_pair(x,cost)); 
	}

	void make_edgedir(int x,int y,int cost){ 
		e[x].push_back(make_pair(y,cost)); 
	}

	void calcdistance(){ 
		// <-cost,x>
		//cost startv->x
		priority_queue<pair<int,int> > q;
		pair<int,int> p;
		vector<bool> did;
		for(int i=0;i<v;i++)did.push_back(false); 
		did[startv]=true;
		for(int i=0;i<e[startv].size();i++){
			p.first=-e[startv][i].second;
			p.second=e[startv][i].first;
			q.push(p);
		}
		while(q.size()!=0){
			pair<int,int> p;
			p=q.top();
			q.pop();
			int x,costsx;
			x=p.second;
			costsx=-p.first;
			if(did[x]==true)continue;
			did[x]=true;
			d[x]=costsx;
			for(int i=0;i<e[x].size();i++){
				p.first=-(d[x]+e[x][i].second);
				p.second=e[x][i].first;
				if(did[p.second]==false)q.push(p);
			}
		}
	}
};
////////////333/////////////444//////////////////////////////////
struct topolodical{
	//0-indexed   <---important
	//ans.size()==v -> heiro nasi
	ll v,e,numsort;
	vector<vector<ll> > g;
	vector<bool> used;
	vector<ll> ans;
	topolodical(ll a,ll b){
		v=a;e=b;
		used.resize(v);
		g.resize(v);
	}
	void make_edgedir(ll s,ll t){//s->t
		g[s].push_back(t);
	}
	void dfs(ll u){
		if(used[u]) return;
		used[u] = true;
		for(auto& i:g[u])dfs(i);
		ans.push_back(u);
	}
	void tsort(){
		for(ll i=0;i<v;++i)dfs(i);
		reverse(ans.begin(),ans.end());
	}
	void calcnumsort(){
		vector<ll> dp,a;
		dp.resize(1<<v,0);
		a.resize(v);
		FOR(i,0,v){
			FOR(j,0,g[i].size()){
				ll x = i,y = g[i][j];
				a[y] |= 1<<x;
			}
		}
		dp[0]=1;
		for(ll i=1; i < (1<<v); i++){
			for(ll j=0;j<v;j++){
				if((i & 1<<j) && ((i | a[j])==i)){
					dp[i] += dp[i-(1<<j)];
				}
			}
		}
		numsort=dp[(1<<v)-1];
	}
};
////////////444/////////////555/////////////////////////////
class comb{
	vector<ll> f,fr;
	ll MOD_;
	public:
	//a^(p-1) = 1 (mod p)(p->Prime numbers)
	//a^(p-2) = a^(-1)

	void init(ll n,ll mod){//input max_n
		MOD_ = mod;
		f.resize(n+1);
		fr.resize(n+1);
		f[0]=fr[0]=1;
		for(ll i=1;i<n+1;i++){
			f[i] = (f[i-1] * i) % mod;
		}
		fr[n] = calc(f[n],mod-2,mod);
		for(ll i=n-1;i>=0;i--){
			fr[i] = fr[i+1] * (i+1) % mod;
		}
	}
	ll nCr(ll n,ll r){
		if(n<0||r<0||n<r)return 0;
		return f[n] * fr[r] % MOD_ * fr[n-r] % MOD_;
	}//nHr = n+r-1Cr
};
/////////////////55555/////////////6666666//////////////////////
// http://judge.u-aizu.ac.jp/onlinejudge/review.jsp?rid=2916862#1
struct edge{int to,cap,rev;};
struct Maxflow{//ant book p.194
	vector<vector<edge> > G;
	vector<int> level,iter;
	void initsize(int nv){
		G.resize(nv);
		level.resize(nv);
		iter.resize(nv);
	}
	void add_edge(int from,int to,int cap){
		G[from].push_back((edge){to,cap,(int)G[to].size()});
		G[to].push_back((edge){from,0,(int)G[from].size()-1});
	}
	void bfs(int s){
		fill(level.begin(),level.end(),-1);
		//memset(level,-1,sizeof(level));
		queue<int> que;
		level[s]=0;
		que.push(s);
		while( !que.empty() ){
			int v = que.front();que.pop();
			for(int i=0;i<G[v].size();i++){
				edge &e = G[v][i];
				if(e.cap > 0 && level[e.to]<0){
					level[e.to] = level[v] + 1;
					que.push(e.to);
				}
			}
		}
	}
	int dfs(int v,int t,int f){
		if(v==t)return f;
		for(int &i = iter[v];i<G[v].size();i++){
			edge &e = G[v][i];
			if(e.cap>0 && level[v]<level[e.to]){
				int d = dfs(e.to,t,min(f,e.cap));
				if(d>0){
					e.cap -= d;
					G[e.to][e.rev].cap += d;
					return d;
				}
			}
		}
		return 0;
	}
	int max_flow(int s,int t){
		int flow=0;
		for(;;){
			bfs(s);
			if(level[t]<0)return flow;
			fill(iter.begin(),iter.end(),0);
			//memset(iter,0,sizeof(iter));
			int f;
			while((f=dfs(s,t,INT_MAX))>0){
				flow += f;
			}
		}
	}
};
//////////666//////AOJ/Course/Graph Algorithms//////777777////
// https://atcoder.jp/contests/soundhound2018/submissions/3881273
struct edge{int to,cap,cost,rev;};
struct MinimumCostflow{//ant book p.200
	int V;
	vector<vector<edge> > G;
	vector<int> prevv,preve,dist;
 
	void initsize(int nv){
		G.resize(nv);
		preve.resize(nv);
		prevv.resize(nv);
		dist.resize(nv);
		V=nv;
	}
	void add_edge(int from,int to,int cap,int cost){
		G[from].push_back((edge){to,cap,cost,G[to].size()});
		G[to].push_back((edge){from,0,-cost,G[from].size()-1});
	}
	int min_cost_flow(int s,int t,int f){
		int res = 0;
		while(f>0){
			fill(dist.begin(),dist.end(),INT_MAX);
			dist[s] = 0;
			bool update=true;
			while(update){
				update = false;
				for(int v=0;v<V;v++){
					if(dist[v]==INT_MAX)continue;
					for(int i=0;i<G[v].size();i++){
						edge &e = G[v][i];
						if(e.cap>0&&dist[e.to]>dist[v]+e.cost){
							dist[e.to] = dist[v]+e.cost;
							prevv[e.to]=v;
							preve[e.to]=i;
							update = true;
						}
					}
				}
			}
		 
			if(dist[t]==INT_MAX)return -1;
			int d=f;
			for(int v=t;v!=s;v=prevv[v]){
				d = min(d,G[prevv[v]][preve[v]].cap);
			}
			f -=d;
			res += d*dist[t];
			for(int v=t;v!=s;v=prevv[v]){
				edge &e = G[prevv[v]][preve[v]];
				e.cap -= d;
				G[v][e.rev].cap += d;
			}
		}
		return res;
	}
};
/////////////777//////////////////88888///////////////////////
struct Inversions{// tentosu
	// after init,u can get number of inversions num_inversion(A)
	vector<ll> L,R;
	void initsize(ll n){//input size of A
		L.resize(n/2+2);
		R.resize(n/2+2);
	}
	ll merge(vector<ll> &A,ll n,ll left,ll mid,ll right){
		ll i,j,k;
		ll cnt = 0;
		ll n1 = mid - left;
		ll n2 = right - mid;
		for(i=0;i<n1;i++)L[i]=A[left+i];
		for(i=0;i<n2;i++)R[i]=A[mid+i];
		L[n1]=R[n2]=INT_MAX;//
		i=j=0;
		for(k=left;k<right;k++){
			if(L[i]<=R[j]){
				A[k]=L[i++];
			}else{
				A[k]=R[j++];
				cnt += n1-i;
			}
		}
		return cnt;
	}
	ll mergeSort(vector<ll> &A,ll n,ll left,ll right){
		ll mid;
		ll v1,v2,v3;
		if(left+1<right){
			mid = (left+right)/2;
			v1 = mergeSort(A,n,left,mid);
			v2 = mergeSort(A,n,mid,right);
			v3 = merge(A,n,left,mid,right);
			return v1+v2+v3;
		}else return 0;
	}
	ll num_inversion(vector<ll> A){
		vector<ll> B=A;//Aの順番を変えたくないなら
		return mergeSort(B,B.size(),0,B.size());
	}
};
////////////88888888888888////////9999999999999///////BIT
// http://judge.u-aizu.ac.jp/onlinejudge/review.jsp?rid=3165304#1
class BIT{
public:
	vector<int> bit;
	int M;
	// 1-index
	BIT(int M):
		bit(vector<int>(M+1, 0)), M(M) {}

	int sum(int i) {
		if (!i) return 0;
		return bit[i] + sum(i-(i&-i));
	}

	void add(int i, int x) {
		if (i > M) return;
		bit[i] += x;
		add(i+(i&-i), x);
	}
};
//9999999///////////////1010101010101010/kosokufurie///////////
struct dftclass{
	//https://beta.atcoder.jp/contests/atc001
	using C = complex<double>;
	using VC = vector<C>;
	int dicidesize(int n){
		int m = 1 << 32 - __builtin_clz(2 * n - 2);
		return m;
	}
	void dft(VC &f, double inv = 1.0){
		int n = f.size();
		if(n == 1)return;
		VC f0(n/2),f1(n/2);
		for(int i=0;2*i<n;i++){
			f0[i] = f[2*i];
			f1[i] = f[2*i+1];
		}
		dft(f0,inv);dft(f1,inv);
		C xi_i=1, xi = polar(1.0,inv*2*acos(-1.0)/ n);
		int m = n/2 - 1;
		for(int i= 0;i<n;i++){
			f[i] = f0[i&m] + xi_i * f1[i&m];
			xi_i *= xi;
		}
	}
	void inv_dft(VC &f_){
		dft(f_,-1.0);
		C n = C(f_.size());
		for(auto&& i:f_)i /= n;
	}
	vector<int> ans(VC &f,VC &g,int n){
		// n -> detasize
		vector<int> v(2*n-1);
		dft(f);dft(g);
		int m = dicidesize(n);
		for(int i=0;i<m;i++)f[i]*=g[i];
		inv_dft(f);
		for(int i=0;i<2*n-1;++i){
			v[i] = (int)(f[i].real()+0.5);
		}
		return v;
	} //kosokufurie
};
//////101010101010/////////////////////////11111111//////////
//nap some ok
//http://judge.u-aizu.ac.jp/onlinejudge/review.jsp?rid=3026687#1
struct nap{
	vector<ll> v,w;
	ll N,W;
	vector< vector<ll> > dp;
	nap(ll n,ll wmax){
		N=n;W=wmax;
		v.resize(n);
		w.resize(n);
		vector<ll> wv(W+1,0);
		FOR(i,0,n+1){
			dp.push_back(wv);
		}
	}
	void calc(){
		FOR(i,0,N){
			FOR(j,0,W+1){
				if(j<w[i]){
					dp[i+1][j]=dp[i][j];
				}else{
					dp[i+1][j]=max(dp[i][j],dp[i+1][j-w[i]]+v[i]);
				}
			}
		}
	}
	ll maxval(ll w){
		return dp[N][w];
	}
};
///////11 11 11  11 111    11 //////////////////1212121212
template <typename T>
struct twodimBIT
{
// 1-indexed
// http://judge.u-aizu.ac.jp/onlinejudge/review.jsp?rid=3095512#1
// https://www.slideshare.net/hcpc_hokudai/binary-indexed-tree
private:
	vector< vector<T> > array;
	const int n;
	const int m;

public:
	twodimBIT(int _n,int _m):
		array(_n+1,vector<T>(_m+1,0)), n(_n),m(_m){}

	// (1,1) ~ (x,y) ruisekiwa
	T sum(int x,int y) {
		T s = 0;
		for(int i=x;i>0;i-=i&(-i))
			for(int j=y;j>0;j-=j&(-j))
				s += array[i][j];
		return s;
	}

	// [(x1,y1),(x2,y2)] no souwa (x1<x2,y1<y2)
	T sum(int x1,int y1,int x2,int y2){
		return sum(x2,y2) - sum(x1-1,y2) - sum(x2,y1-1) + sum(x1-1,y1-1);
	}

	void add(int x,int y,T k) {
		for(int i=x;i<=n;i+=i&(-i))
			for(int j=y;j<=m;j+=j&(-j))
				array[i][j] += k;
	}
};
//////////////12    12   12  //////////////13  13  13

// https://onlinejudge.u-aizu.ac.jp/status/users/fullhouse1987/submissions/1/DPL_1_G/judge/3125274/C++
// http://tsutaj.hatenablog.com/entry/2017/01/25/161050
struct KnapWithLimitations{
	ll N,W;
	vector<ll> v,w,m,dp;
	KnapWithLimitations(ll n,ll ww){
		N=n;W=ww;
		v.resize(N);
		w.resize(N);
		m.resize(N);
		dp.resize(W+1,0);
	}
	ll maxvalue(){
		FOR(i,0,N){
			for(ll k=0;m[i]>0;k++){
				ll key = min(m[i],(ll)(1<<k));
				m[i] -= key;
				for(ll j=W;j>=key*w[i];j--){
					dp[j] = max(dp[j],dp[j-key*w[i]]+key*v[i]);
				}
			}
		}
		ll ans = 0;
		FOR(i,0,W+1)ans=max(ans,dp[i]);
		return ans;
	}
};
//////////////////////////131313  ///////////////////14141414
// http://sesenosannko.hatenablog.com/entry/2017/09/01/214852
// https://beta.atcoder.jp/contests/abc061/submissions/3322496
// http://judge.u-aizu.ac.jp/onlinejudge/review.jsp?rid=3166760#1
// https://atcoder.jp/contests/abc137/submissions/6844499
// 0-index
struct Bellman_Ford{
	struct edge{ll from, to, cost;};
	ll inf = 1e16;
	vector<edge> es;
	vector<ll> d;
	ll V,E;
	void init(ll v,ll e){
		V = v; E = e;
		d.resize(V);
	}
	void shortest_path(ll s){
		FOR(i,0,V) d[i] = inf;
		d[s] = 0;
		FOR(i,0,V){
			FOR(j,0,E){
				edge e = es[j];
				if(d[e.from]!=inf && d[e.to]>d[e.from] + e.cost){
					d[e.to] = d[e.from] + e.cost;
				}
			}
		}
	}
	bool find_negative_loop(){// all graph
		FOR(i,0,V)d[i] = 0;
		FOR(i,0,V){
			FOR(j,0,E){
				edge e = es[j];
				if(d[e.to]>d[e.from] + e.cost){
					d[e.to] = d[e.from] + e.cost;
					if(i == V-1) return true;
				}
			}
		}
		return false;
	}
	bool find_negative_loop(ll s){// from s
		ll cnt = 0;
		FOR(i,0,V)d[i]=inf;
		d[s]=0;
		while(true){
			bool update = false;
			cnt++;
			FOR(i,0,E){
				edge e = es[i];
				if(d[e.from] != inf && d[e.to]>d[e.from] + e.cost){
					d[e.to] = d[e.from] + e.cost;
					if(cnt == V){
						return true;
					}
					update = true;
				}
			}
			if(! update)break;
		}
		return false;
	}
	bool shortest_path(int s, int t){ // t: destination
		FOR(i,0,V)d[i] = inf;
		d[s] = 0;
		FOR(i,0,2*V){
			FOR(j,0,E){
				edge e = es[j];
				if(d[e.from]!=inf && d[e.to]>d[e.from] + e.cost){
					d[e.to] = d[e.from] + e.cost;
					if(i>=V-1 && e.to==t) return true;
				}
			}
		} 
		return false;
	}
};
// ///////////////////////141414141414141414     14  15151515

// http://judge.u-aizu.ac.jp/onlinejudge/review.jsp?rid=3178606#1
// ant p.286
struct SCC{// Strongly Connected Components
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

// ///////////151515151515151515151515   1616161616161616

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
/// 161616161616161616  17171717171717
// https://atcoder.jp/contests/colopl2018-final/submissions/4341467
// https://atcoder.jp/contests/colopl2018-final/submissions/4364468
template <typename T> class ConvexHullTrick{
public:
	deque<pair<T,T>> lines;
	bool is_needless(pair<T,T> a, pair<T,T> b, pair<T,T> c){
		return (a.second-b.second)*(a.first-c.first) >= (a.second-c.second)*(a.first-b.first);
	}
	void add(T a, T b){
		if(!lines.empty()){
			auto l = lines.back();
			if(l.first == a){
				if(l.second < b) return;
				else lines.pop_back();
			}
		}
		while(lines.size()>=2 && is_needless(make_pair(a,b), lines.back(), *(lines.end()-2))){
			lines.pop_back();
		}
		lines.push_back(make_pair(a,b));
	}
	T apply(pair<T,T> f, T x){
		return f.first*x + f.second;
	}
	T query(T x){
		while(lines.size()>=2 && apply(lines[0],x)>apply(lines[1],x)){
			lines.pop_front();
		}
		return apply(lines[0],x);
	}
};
/// 17 17 17 17 17 17     buguhiuhiuhiuhkuygduysgduyfg             18 18 18 18
//無向グラフ
// http://kagamiz.hatenablog.com/entry/2013/10/05/005213
// http://judge.u-aizu.ac.jp/onlinejudge/review.jsp?rid=3424140
// http://judge.u-aizu.ac.jp/onlinejudge/review.jsp?rid=3424144#1
struct Articulation_point_and_Bridge{
int V,E=0;
vector< vector<int> > G;
vector<int> ord,low;
vector<bool> vis;
vector< pair<int, int> > bridge;
vector<int> articulation;
Articulation_point_and_Bridge(int V_){
	V=V_;
	G.resize(V);
	ord.resize(V,0);
	low.resize(V,0);
	vis.resize(V,false);
}
void dfs(int v, int p, int &k){
	vis[v] = true;
	ord[v] = k++;
	low[v] = ord[v];
	bool isArticulation = false;
	int ct = 0;
	for (int i = 0; i < G[v].size(); i++){
		if (!vis[G[v][i]]){
			ct++;
			dfs(G[v][i], v, k);
			low[v] = min(low[v], low[G[v][i]]);
			if (~p && ord[v] <= low[G[v][i]]) isArticulation = true;
			if (ord[v] < low[G[v][i]]) bridge.push_back(make_pair(min(v, G[v][i]), max(v, G[v][i])));
		}
		else if (G[v][i] != p){
			low[v] = min(low[v], ord[G[v][i]]);
		}
	}
	
	if (p == -1 && ct > 1) isArticulation = true;
	if (isArticulation) articulation.push_back(v);
}
void add_edge(int v1,int v2){
	G[v1].push_back(v2);
	G[v2].push_back(v1);
	E++;
}
void calc(){
	int k=0;
	for(int i=0;i<V;i++){
		if(!vis[i])dfs(i,-1,k);
	}
	sort(bridge.begin(), bridge.end());
	sort(articulation.begin(), articulation.end());
}

};
//////////////////////18181818181818             19 19 asdaksndaojsndo           19
// http://judge.u-aizu.ac.jp/onlinejudge/review.jsp?rid=3424270
// http://techtipshoge.blogspot.com/2016/08/blog-post.html
// http://judge.u-aizu.ac.jp/onlinejudge/review.jsp?rid=4108283#1 全方位木DPで？
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
// 19        19             19           20           20    20  20      20  20
// https://atcoder.jp/contests/abc036/submissions/6785659
template <typename T>
struct zatsu{
	map<T,T> re;// zatsu -> real
	map<T,T> za;// real -> zatsu
	zatsu(vector<T> ini){make(ini);}
	void make(vector<T> x){
		re.clear();
		za.clear();
		sort(x.begin(), x.end());
		x.erase(unique(x.begin(),x.end()),x.end());
		ll z = 0;
		for(auto a:x){
			za[a] = z;
			re[z] = a;
			z++;
		}
	}
};
// https://atcoder.jp/contests/abc036/submissions/6786299
template <typename T>
struct zatsu{
	vector<T> x;
	zatsu(vector<T> ini){make(ini);}
	void make(vector<T> X){
		x = X;
		sort(x.begin(), x.end());
		x.erase(unique(x.begin(),x.end()),x.end());
	}
	T change(T X){
		return (lower_bound(x.begin(),x.end(),X)-x.begin());
	}
};
//////2020202020202020202020221212121212121212121212121212121212
struct RollingHash{
	ll M;
	vector<ll> H,B;
	RollingHash(string s,ll b=1009LL,ll m=1000000007LL){
		//ll b= 9973LL,ll m = 999999937LL
		M = m;
		ll n = s.size();
		B.resize(n+1);
		H.resize(n+1,0);
		B[0] = 1;
		for(ll i=0;i<n;i++){
			B[i+1] = (B[i]*b)%M;
			H[i+1] = (H[i]*b%M+s[i])%M;
		}
	}
	ll GetHash(ll l,ll r){ // [l,r)
		return (H[r]+(M-B[r-l]*H[l]%M))%M;
	}
};
////////////////////
template <typename T>
class RSQ{
public:
	vector<T> dat;
	int n;
	RSQ(int n0,T initdata){
		initsize(n0,initdata);
	}
	void initsize(int n0, T initdata){
		int k=1;
		while(1){
			if(n0<=k){
				n=k;
				dat.resize(2*n-1);
				for(int i=0;i<2*n-1;i++)dat[i]=initdata;
				break;
			}
			k*=2;
		}
	}

	void add(int i,int x){
		i += n-1;
		dat[i] += x;
		while(i>0){
			i = (i-1) / 2;
			dat[i] = dat[i*2+1]+dat[i*2+2];
		}
	}

	T query0(int a,int b,int k,int l,int r){// [a,b)
		if(r<=a || b<=l)return 0;
		if(a<=l && r<=b)return dat[k];
		else{
			int vl = query0(a,b,k*2+1,l,(l+r)/2);
			int vr = query0(a,b,k*2+2,(l+r)/2,r);
			return vl+vr;
		}
	}

	T query(int a,int b){ // [a,b)
		return query0(a,b,0,0,n);
	}
};
/////////////////////////////////232323/////////////////////////23
//https://onlinejudge.u-aizu.ac.jp/solutions/problem/DSL_2_I/review/3810661/fullhouse1987/C++14 //RSQ and RUQ
//https://onlinejudge.u-aizu.ac.jp/status/users/fullhouse1987/submissions/14/DSL_2_D/judge/3812299/C++14
struct RUQ{//平方分割 0-index
	int n,rn,bn,Q=0;
	vector<int> a,b,at,bt;
	RUQ(int n_){
		n = n_; rn = sqrt(n); bn = ceil((double)n/rn);
		a.resize(n,2147483647);at.resize(n,0);
		b.resize(bn,2147483647);bt.resize(n,0);
	}
	int value(int s){
		return (at[s]<bt[s/rn]?b[s/rn]:a[s]);
	}
	void updete(int s,int t,int x){
		Q++;
		FOR(i,s,t+1){
			if(i%rn==0 && (i+rn-1)<=t){
				b[i/rn] = x;
				bt[i/rn] = Q;
				i += rn-1;
			}else{
				a[i] = x;
				at[i] = Q;
			}
		}
	}
};
//24         24             24

// http://judge.u-aizu.ac.jp/onlinejudge/review.jsp?rid=3810628#1
// 区間加算　一点取得 RAQ 平方分割で実装
template <typename T>
struct RAQ{// 0-index
	int n,rn,bn; // rn個のかたまりをbn個
	vector<T> data,bucket;
	RAQ(int n_){init(n_);}
	void init(int n_){
		n = n_;
		rn = sqrt(n);
		bn = ceil((double)n/rn);
		data.resize(n,0);
		bucket.resize(bn,0);
	}
	T get(int i){
		return data[i]+bucket[i/rn];
	}
	void add(int s,int t,T x){ // s~tにxを加算
		for(int i=s;i<=t;++i){
			if(i%rn==0 && (i+rn-1)<=t){
				bucket[i/rn] += x;
				i += rn-1;
			}else{
				data[i] += x;
			}
		}
	}
};


///////////////////////////////24
// http://judge.u-aizu.ac.jp/onlinejudge/review.jsp?rid=4383376#1 add min
// http://judge.u-aizu.ac.jp/onlinejudge/review.jsp?rid=4383454#1 change min
// https://yukicoder.me/submissions/693392  add sum
// https://onlinejudge.u-aizu.ac.jp/solutions/problem/GRL_5_E/review/5830132/fullhouse1987/C++11  add sum
// https://atcoder.jp/contests/abc216/submissions/25506009 change sum
// https://atcoder.jp/contests/abl/submissions/25632299 change 10^(L-k)*d[k] マージミスってるから区間が丁度のところのみ合ってる  
// https://atcoder.jp/contests/practice2/submissions/25699849  区間のaをa*b+cに変換&区間和
template <typename T>
class Lazy_Seg_Tree{
public: // 0-index
	// buketは区間変更の値  datは区間の答え
	// initial_d,initial_b は buket dat の 単位元
	T make_dat_from_buket(T da, T bu, T s){//dat[k] = make_dat_from_buket(dat[k],buket[k],r-l);//遅延配列から値配列を作る
		return da + bu;
	}
	T buket_to_child(T ch, T pa){// buket[2*k+1] = buket_to_child(buket[2*k+1],buket[k]);//遅延配列で子へ伝播
		return ch + pa;
	}
	T buket_update(T bu , T x , T s){// buket[k] = buket_update(buket[k],x,r-l); //遅延配列を作る
		return bu + x;
	}
	T dat_update(T dc1, T dc2){// dat[k] = dat_update(dat[2*k+1],dat[2*k+2]); //2つの子から値配列を作る
		return min(dc1 , dc2);
	}

	vector<T> dat,buket;
	T initial_d,initial_b,M;
	int n;
	Lazy_Seg_Tree(int n0_=1,T initial_dat=1,T initial_buket=0 ,T M_=1){
		initsize(n0_,initial_dat,initial_buket,M_);
	}
	void initsize(int n0,T initial_da,T initial_bu,T M_){
		M = M_;
		initial_d = initial_da;
		initial_b = initial_bu;
		int k=1;
		while(k<n0)k*=2;
		n = k;
		dat.resize(2*n-1);
		buket.resize(2*n-1);
		for(int i=0;i<2*n-1;i++)dat[i]=initial_da;
		for(int i=0;i<2*n-1;i++)buket[i]=initial_bu;
	}
	
	void eval(int k,int l,int r){
		if(buket[k] != initial_b){
			dat[k] = make_dat_from_buket(dat[k],buket[k],r-l);
			if(r-l>1){
				buket[2*k+1] = buket_to_child(buket[2*k+1],buket[k]);
				buket[2*k+2] = buket_to_child(buket[2*k+2],buket[k]);
			}
			buket[k] = initial_b;
		}
	}

	void operate(int a,int b, T x, int k=0,int l=0, int r=-1){
		if(r<0)r=n;
		eval(k,l,r);
		if(b<=l || r<=a)return;
		if(a<=l && r<=b){
			buket[k] = buket_update(buket[k],x,r-l);
			eval(k,l,r);
		}else{
			operate(a,b,x,2*k+1,l,(l+r)/2);
			operate(a,b,x,2*k+2,(l+r)/2,r);
			dat[k] = dat_update(dat[2*k+1],dat[2*k+2]);
		}
	}
	T get(int a,int b,int k=0,int l=0,int r=-1){
		if(r<0)r=n;
		if(b<=l || r<=a) return initial_d;
		eval(k,l,r);
		if(a<=l && r<=b) return dat[k];
		T vl = get(a,b,2*k+1,l,(l+r)/2);
		T vr = get(a,b,2*k+2,(l+r)/2,r);
		return dat_update(vl,vr);
	}
};

/////////////////////////////////////////////////////////////////25

///////////////////////////////////////29//29/////////////////////29
//29 最近点対 https://onlinejudge.u-aizu.ac.jp/status/users/fullhouse1987/submissions/1/CGL_5_A/judge/5789282/C++14
double Closest_Pair(vector<pair<double,double>> X){
	sort(X.begin(), X.end());
	double EPS = 1e-7;
	if(X.size() < 11){
		double ans = 1e18;
		for(int i=0;i<X.size();i++){
			for(int j=i+1;j<X.size();j++){
				double dx = (X[i].first-X[j].first);
				double dy = (X[i].second-X[j].second);
				ans = min(ans,sqrt(dx*dx+dy*dy));
			}
		}
		return ans;
	} 
	vector<pair<double,double>> Y,Z;
	int ny = X.size() / 2;
	int nz = X.size() - ny;
	for(int i=0; i < ny ;i++) Y.push_back(X[i]);
	for(int i=ny;i<ny+nz;i++) Z.push_back(X[i]);
	double d = min( Closest_Pair(Y) , Closest_Pair(Z) );
	double xc = Z[0].first;
	vector<pair<double,double>> yx;
	for(int i=0;i<ny;i++) if( fabs(Y[i].first-xc) <= d + EPS) yx.push_back({ Y[i].second , Y[i].first });
	for(int i=0;i<nz;i++) if( fabs(Z[i].first-xc) <= d + EPS) yx.push_back({ Z[i].second , Z[i].first });
	sort(yx.begin(), yx.end());
	double res = d;
	for(int i=0;i<yx.size();i++){
		int j = i+1;
		while( j<yx.size() && fabs(yx[j].first-yx[i].first) <= d + EPS){
			double dx = (yx[i].first-yx[j].first);
			double dy = (yx[i].second-yx[j].second);
			res = min(res,sqrt(dx*dx+dy*dy));
			j++;
		}
	}
	return res;
}
/////////2929///////////30////30///////////////////////30//////////////30//////////////////////30
// 列に対してできる操作を木のパスについても出来るようにするやつ
// https://yukicoder.me/submissions/693392 
// https://judge.u-aizu.ac.jp/onlinejudge/review.jsp?rid=5829941#1    LCAあり hl.HLvector[hl.index_LCA]
// https://atcoder.jp/contests/abc133/submissions/25368840
// https://onlinejudge.u-aizu.ac.jp/solutions/problem/GRL_5_E/review/5830132/fullhouse1987/C++11
// https://www.hackerrank.com/challenges/subtrees-and-paths/submissions/code/230708232 部分木もOK (https://www.hackerrank.com/challenges/subtrees-and-paths/submissions/code/230707758)
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
		vector< pair<int,int> > res;// secondも含む
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
	pair<int,int> subtree_index(int i){ // HLvectorで頂点iの部分木のインデックスの両端 res.secondは含まない
		return {index[i],index[i]+num[i]}; // [res.first,res.second-1]が部分木
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

////  31   31 ///////////////////////////////////31//////////////////////////
// https://atcoder.jp/contests/code-thanks-festival-2017/submissions/25390046
// けんちょんの解説  https://drken1215.hatenablog.com/entry/2019/03/20/202800
struct Gauss_Jordan_elimination_mod2{
	int N,M,mod=2,free; // 行列 N行M列 
	vector< vector<int> > A;
	vector< int > B;
	Gauss_Jordan_elimination_mod2(int N,int M):
		A(vector<vector<int>>(N,vector<int>(M,0))), B(vector<int>(N,0)), N(N) , M(M) {}
	bool set_retsu(int J,vector<int> X){ // J列目を作る
		if((int)X.size() != N) return false;
		if(J >= M) return false;
		for(int i=0;i<N;i++) A[i][J] = X[i] % mod;
		return true;
	}
	bool set_gyo(int I,vector<int> X){ // I行目を作る
		if((int)X.size() != M) return false;
		if(I >= N) return false;
		for(int j=0;j<M;j++) A[I][j] = X[j] % mod;
		return true;
	}
	bool set_B(vector<int> X){
		if((int)X.size() != N) return false;
		for(int i=0;i<N;i++) B[i] = X[i] % mod;
		return true;
	}
 
	bool calc(){ // Ax = B となるxを探す！
		int ok = 0; // ok個1確定した
		free = 0;
		for(int x=0;x<M;x++){ //x列を見る
			int i = -1;
			for(int j=ok;j<N;j++) if(A[j][x]) i = j;
			if(i < 0){
				free++;
				continue;
			}
			for(int j=x;j<M;j++)swap(A[i][j],A[ok][j]);
			swap(B[ok],B[i]);
			for(int j=ok+1;j<N;j++){
				if(A[j][x]){
					for(int k=x;k<M;k++){
						(A[j][k] += A[ok][k]) %= mod;
					}
					(B[j] += B[ok]) %= mod;
				}
			}
			ok++;
			//if(ok==N)break;
		}
		for(int i=ok;i<N;i++){
			if(B[i] && count(A[i].begin(), A[i].end(),1)==0){
				free = -1;
				return false;
			}
		}
		return true;
	}
};
///////////////////////////////////////////////////////////////////////////////////
