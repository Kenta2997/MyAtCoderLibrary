// https://atcoder.jp/contests/abc087/submissions/3834180

class UnionFind{
public:
	vector<ll> par;
	vector<ll> d;
	//0-indexed
	UnionFind(ll n){
		init(n);
	}
	void init(ll n = 1) {
		par.resize(n);
		d.resize(n,0);
		for (ll i = 0; i < n; ++i) par[i] = i;
	}
	ll len1(ll x){
		if(par[x]==x)return 0;
		ll ans = d[x];
		ans += len1(par[x]);
		return ans;
	}
	ll root(ll x = 1){
		if(par[x]==x)return x;
		else{
			d[x] = len1(par[x]) + d[x];
			return par[x] = root(par[x]);
		}
	}
	ll length(ll x,ll y){ // xから左にlengthいったところにy
		return len1(y) - len1(x);
	}
	void unite(ll x,ll y,ll dd){ // xから左にdだけ離れたところにy
		dd += len1(x);x=root(x);
		dd -= len1(y);y=root(y);
		if(x==y)return;
		if(dd>=0){
			d[y] = dd;
			par[y] = x;
		}else{
			d[x] = -dd;
			par[x] = y;
		}
	}
};