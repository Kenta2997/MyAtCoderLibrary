// コンビネーション列挙
// http://judge.u-aizu.ac.jp/onlinejudge/review.jsp?rid=3454158#1

// n個の中からk個選ぶ 0~n-1 bitで表現 n<=18 k<=n
vector<ll> combination(ll n,ll k){
	vector<ll> ans;
	ll x = (1LL<<k)-1;
	ll f = x<<(n-k);
	while(1){
		ans.push_back(x);
		if(x==f)return ans;
		x += x&-x;
		ll pcount = bitset<64>(x).count();
		x|=(1LL<<(k-pcount))-1;
	}
}

// 集合xの次
ll bits(ll x,ll k){
	ll fb = x&-x;
	x += fb;
	ll pcount=bitset<64>(x).count();
	x|=(1LL<<(k-pcount))-1;
	return x;
}