// http://judge.u-aizu.ac.jp/onlinejudge/review.jsp?rid=3818388
struct RollingHash{
	ll M;
	vector<ll> H,B;
	RollingHash(string s,ll b=1009LL,ll m=1000000007LL){
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
