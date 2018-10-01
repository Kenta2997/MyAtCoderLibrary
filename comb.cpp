class comb{
	vector<ll> f,fr;
	ll MOD;
	public:
	//a^(p-1) = 1 (mod p)(p->Prime numbers)
	//a^(p-2) = a^(-1)
	ll calc(ll a,ll b,ll p){//a^(b) mod p   
		if(b==0)return 1;
		ll y = calc(a,b/2,p);y=(y*y)%p;
		if(b & 1) y = (y * a) % p;
		return y;
	}
	void init(ll n,ll mod){//input max_n
		MOD = mod;
		f.resize(n+1);
		fr.resize(n+1);
		f[0]=fr[0]=1;
		for(ll i=1;i<n+1;i++){
			f[i] = (f[i-1] * i) % mod;
			fr[i] = calc(f[i],mod-2,mod);
		}
	}
	ll nCr(ll n,ll r){
		return f[n] * fr[r] % MOD * fr[n-r] % MOD;
	}
};