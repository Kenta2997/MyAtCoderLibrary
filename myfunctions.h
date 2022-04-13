//1 Levenshtein -> hensyukyori
//2 LIS Longest Increasing Subsequence
//3 LCS Longest Common Subsequence
//4 divisor_sums no vali
//5 ax+by=gcd(a,b)
//6 soinsubunkai 
//7 約数列挙
//8 コンビネーション列挙
//9 下(上)からi個の点を1つにまとめた時の最小移動距離 https://yukicoder.me/submissions/352391
//10 set iterater https://yukicoder.me/submissions/373976
//11 篩
//12 木で、ある頂点を除いた時にできる森の各木のサイズを出力 O(NlogN)だと思う
//13 leaf
//14 木で葉から根に向かう順に辺を列挙
//15 2部グラフ判定
//16 1D当たり判定　端o---o---o 白丸
//17 正方行列累乗 https://yukicoder.me/submissions/479363
//18 グラフbfsで最短距離
//19 boost の float 例　https://atcoder.jp/contests/abc191/submissions/20007670
//20 boost 多倍長 https://boostjp.github.io/tips/multiprec-int.html  
//21 乱数　	mt19937_64 mt(seed);uniform_real_distribution<double> rnd(0.00,1.00);
//22 ファイル操作　#include <fstream>   std::ifstream ifs(fileName); std::ofstream ofs(fileName); 
//23 高速化  https://atcoder.jp/contests/abc091/submissions/21210493 , https://atcoder.jp/contests/arc021/submissions/29562679?lang=ja
//24 constexpr https://atcoder.jp/contests/abc154/submissions/10025879
//25 桁DP　https://atcoder.jp/contests/abc154/submissions/9999866　
//26 マラソンレート
//27 行列累乗 3×3 https://yukicoder.me/submissions/676734
////////111//////////////////////////////////////////////////////
int Levenshtein(string s1,string s2){//hensyu kyori
	vector< vector<int> > dp;
	int ss1=s1.size(),ss2=s2.size();
	int n=max(ss1,ss2)+1;
	dp.resize(n);for(int i=0;i<n;i++)dp[i].resize(n);
	for(int i=0;i<ss1+1;i++)dp[i][0]=i;
	for(int i=0;i<ss2+1;i++)dp[0][i]=i;
	for(int i=1;i<ss1+1;i++){
		for(int j=1;j<ss2+1;j++){
			int a,b=1;
			if(s1[i-1]==s2[j-1])b=0;
			a=min(dp[i-1][j]+1,dp[i][j-1]+1);
			a=min(a,dp[i-1][j-1]+b);
			dp[i][j]=a;
		}
	}
	return dp[s1.size()][s2.size()];
}
//////////111//////////////////222////////////////////////////////
int LIS(vector<int> a){
	vector<int> dp;
	const int infi=1e9+7;
	int n = a.size();
	dp.resize(n);
	fill(dp.begin(),dp.end(),infi);
	for(int i=0;i<n;i++){
		*lower_bound(dp.begin(),dp.end(),a[i])=a[i];
	}
	int ans = lower_bound(dp.begin(),dp.end(),infi)-dp.begin();
	return ans;
}
//////////222////////////////////////////3333////////////////////
//http://judge.u-aizu.ac.jp/onlinejudge/review.jsp?rid=3043199
ll LCS(string x,string y){
	ll xs,ys;
	xs=x.size();ys=y.size();
	ll dp[xs+1][ys+1];
	FOR(i,0,xs+1)FOR(j,0,ys+1)dp[i][j]=0;
	FOR(i,0,xs+1){
		FOR(j,0,ys+1){
			if(i==0||j==0)dp[i][j]=0;
			else{
				if(x[i-1]==y[j-1])dp[i][j]=dp[i-1][j-1]+1;
				else dp[i][j] = max(dp[i-1][j],dp[i][j-1]);
			}
		}
	}
	return dp[xs][ys];
}
//////////////333///////////////4444///////////////////////////
ll divisor_sum(ll x){
	ll ans = 1;
	ll p = x;
	for(ll i=2;i<sqrt(p)+1;++i){
		ll kake = 1;
		ll tasi = 1;
		while(x%i==0){
			kake += tasi*i;
			x/=i;
			tasi*=i;
		}
		ans *= kake;
	}
	if(ans!=1)return ans*(x+1);
	else return p+1;
}
///////////////////////////////////////4444    ////////55///////
//https://onlinejudge.u-aizu.ac.jp/status/users/fullhouse1987/submissions/2/NTL_1_E/judge/3125028/C++
// ax+by=gcd(a,b) find(x,y)  (a,b)isConst 
ll ext_gcd(ll a,ll b,ll &x ,ll &y){
	if(b==0){
		x = 1;y =0;return a;
	}
	ll q = a/b;
	ll g = ext_gcd(b,a-q*b,x,y);
	ll z = x - q*y;
	x = y;y = z;
	return g;
}
////////////////////////////////////555555/////////66666//////66666
map<ll,ll> soinsubunkai(ll x){    // x = {first^second}*...
	map<ll,ll> ans;
	if(x==1){ans[1]++;return ans;}
	ll p=x;
	for(ll i=2;i*i<=x;i++){
		while(p%i==0){
			p /= i;
			ans[i]++;
		}
	}
	if(p!=1)ans[p]++;
	return ans;
}

// https://algo-logic.info/prime-fact/
// https://atcoder.jp/contests/abc215/submissions/25310246
// 前処理 O(NloglogN) クエリO(log Q)で N以下のQを素因数分解できる
template <typename T>
struct PrimeFact {
	vector<T> spf;
	PrimeFact(T N) { init(N); }
	void init(T N) { // 前処理。spf を求める
		spf.assign(N + 1, 0);
		for (T i = 0; i <= N; i++) spf[i] = i;
		for (T i = 2; i * i <= N; i++) {
			if (spf[i] == i) {
				for (T j = i * i; j <= N; j += i) {
					if (spf[j] == j) {
						spf[j] = i;
					}
				}
			}
		}
	}
	map<T, T> get(T n) { // nの素因数分解を求める
		map<T, T> m;
		while (n != 1) {
			m[spf[n]]++;
			n /= spf[n];
		}
		return m;
	}
};

////////////////6666666666666666666666666666666///77777777777777777777
vector<ll> enum_div(ll n){//yakusu
	vector<ll> ret;
	for(ll i=1 ; i*i<=n ; ++i){
			if(n%i == 0){
				ret.push_back(i);
				if(i*i!=n)ret.push_back(n/i);
			}
	}
	return ret;
}
///////////////////////////77777777777777777777777//8888888888888888888888888888

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
///////////////////////////88888888888888888888888//9999999999999999999999999999

//11         11     11
vector<int> make_p(int x){
	vector<bool> isp(x+1,true);
	isp[0]=isp[1]=false;
	for(int i=2;i*i<=x;i++){
		if(isp[i]){
			for(int j=2*i;j<=x;j+=i){
				isp[j] = false;
			}
		}
	}
	vector<int> res;
	for(int i=2;i<=x;i++){
		if(isp[i])res.push_back(i);
	}
	return res;
}

//////11 11 11 11 11 11 11 //////////////12 12 12 12 12 12 12 12 12 12 12 12

//関数版   	https://atcoder.jp/contests/abc149/submissions/9245902
//クラス版		https://atcoder.jp/contests/abc149/submissions/9243909
void get_forest_size(const vector<int> &u,const vector<int> &v,vector<vector<int>> &forest){
	// 木で、ある頂点を除いた時にできる森の各木のサイズを出力 0-index
	// out[i][j] := 頂点iを取り除いてできた森のうちj個目の木のサイズ
	// 辺集合が引数
	// vector<vector<int>> &forest -> 出力用
	int N = u.size() + 1;
 
	vector<int> us(N-1,0),vs(N-1,0);
	// us[i] := i番目の辺のv[i]側から見たu[i]側にある頂点数
	// vs[i] := i番目の辺のu[i]側から見たv[i]側にある頂点数
 
	set<int> connect[N];
	vector<int> num(N,0);
	vector<int> did(N-1,0);//1->uでやった 2->vでやった
	for(int i=0;i<N-1;i++){
		connect[u[i]].insert(i);
		connect[v[i]].insert(i);
	}
	queue<int> Q;//辺の番号が入る
	for(int i=0;i<N;++i){
		if(connect[i].size() == 1){
			Q.push( *connect[i].begin() );
		}
	}
	//葉の接続してる辺から取り除いていき、また葉ができら取り除いていく
	while( Q.size() ){
		int ie = Q.front();
		Q.pop();
		if(connect[u[ie]].size() == 1){
			us[ie] = num[u[ie]] + 1;
			num[v[ie]] += us[ie];
			connect[u[ie]].clear();
			connect[v[ie]].erase(ie);
			if(connect[v[ie]].size()==1){
				Q.push( *connect[v[ie]].begin() );
			}
			if(not did[ie]) did[ie] = 1;
		}else{
			vs[ie] = num[v[ie]] + 1;
			num[u[ie]] += vs[ie];
			connect[v[ie]].clear();
			connect[u[ie]].erase(ie);
			if(connect[u[ie]].size()==1){
				Q.push( *connect[u[ie]].begin() );
			}
			if(not did[ie]) did[ie] = 2;
		}
	}
	forest.resize(N);
	for(int i=0;i<N-1;i++){
		if(did[i]==1){
			vs[i] = N - us[i];
		}else{
			us[i] = N - vs[i];
		}
		forest[u[i]].push_back(vs[i]);
		forest[v[i]].push_back(us[i]);
	}
	return;
}
//////////////////12 12 12 12 12 12 12      13 13 13 13 13 13 13 13 13
vector<int> leaf(const vector<int> &u,const vector<int> &v){
	int N = u.size() + 1;
	vector<int> G(N,0),res;
	for(int i=0;i<N-1;i++){
		G[u[i]]++;
		G[v[i]]++;
	}
	for(int i=0;i<N;i++){
		if(G[i] == 1){
			res.push_back(i);
		}
	}
	return res;
}
//////////////////////////////////13131313131313131313 1414141414141414141414
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
////////////////////////////////14             14             15          15
// https://atcoder.jp/contests/code-festival-2017-qualb/submissions/11574055
vector<int> calc(vector<vector<ll>> &G){//2部グラフ判定 res[0]=-1なら2部グラフじゃない
  vector<int> res(G.size(),-1);
  vector<bool> did(G.size(),false);
  int n = G.size();
  for(int i=0;i<n;i++){
    if(did[i])continue;
    res[i] = 0;
    queue<int> q;
    q.push(i);
    did[i] = true;
    while(q.size()){
      int from = q.front();
      q.pop();
      for(auto to:G[from]){
        if(did[to] && res[from]==res[to]){
          res[0] = -1;
          return res;
        }else if(not did[to]){
          q.push(to);
          res[to] = 1 - res[from];
          did[to] = true;
        }
      }
    }
  }
  return res;
}
////////////////////////15          ///////////15///////////16
bool match(P a,P b){
  int c = max(a.first,b.first);
  int d = min(a.second,b.second);
  return c<d;
}
////////////////////////////////16 16 //////////////////////17
// https://yukicoder.me/submissions/479363
vector< vector<ll> > pow_(vector<vector<ll>> A,ll k){
  ll M = A.size();
  vector< vector<ll> > res(M,vector<ll>(M,0));
  if(k==0){
    FOR(i,0,M)res[i][i] = 1;
    return res;
  }
  if(k==1){
    return A;
  }
  auto x = pow_(A,k/2);
  FOR(i,0,M){
    FOR(j,0,M){
      FOR(m,0,M){
        (res[i][j] += x[i][m]*x[m][j]) %=MOD;
      }
    }
  }
  if(k&1){
    FOR(i,0,M)FOR(j,0,M)x[i][j] = 0;
    FOR(i,0,M){
      FOR(j,0,M){
        FOR(m,0,M){
          (x[i][j] += res[i][m]*A[m][j]) %=MOD;
        }
      }
    }
    return x;
  }
  return res;
}
//////////////////17            17 /////////////////18 ////////////18
//https://atcoder.jp/contests/abc168/submissions/13367655
vector<int> dis_(int n,vector<int> &a,vector<int> &b,int start){
  vector<int> G[n];
  FOR(i,0,a.size()){
    G[a[i]].push_back(b[i]);
    G[b[i]].push_back(a[i]);
  }
  vector<int> res(n,-1);
  queue<int> nex;
  nex.push(start);
  res[start] = 0;
  while(nex.size()){
    int from = nex.front();
    nex.pop();
    for(auto to:G[from]){
      if(res[to]>=0)continue;
      res[to] = res[from] + 1;
      nex.push(to);
    }
  }
  return res;
}


/////////////////////////////////////////////////////26 
void rate_marason(vector<int> Paf){
	// パフォ 1989,1591,975,522,1591,1182,2009,1974,1201,1660
	long double S = 724.4744301;
	long double R = 0.8271973364;
	vector<long double> Q,tmp;
	for(auto p:Paf){
		for(int i=1;i<=100;i++){
			Q.push_back( 1.0*p - S * log(1.0*i) );
		}
		sort(Q.begin(), Q.end());
		reverse(Q.begin(), Q.end());
		Q.resize(100);
		long double r = 0;
		for(int i=99;i>=0;i--){
			r = ( r + Q[i] ) * R; 
		}
		cout << round(r*(1.0-R)/(R-pow(R,101))) << endl;
	}
}
/////////////////////////////////////////////////////////////