/*#include <algorithm>
#include <cstdio>
#include <iostream>
#include <map>
#include <cmath>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <bitset>
#include <cstring>
#include <deque>
#include <iomanip>
#include <limits>
#include <fstream>
#include <random>
using namespace std;
typedef long long ll;
//#define P pair<ll,ll>
#define FOR(I,A,B) for(int I = int(A); I < int(B); ++I)
#define FORR(I,A,B) for(int I = int((B)-1); I >= int(A); --I)
#define POSL(x,v) (lower_bound(x.begin(),x.end(),v)-x.begin()) //xi>=v  x is sorted
#define POSU(x,v) (upper_bound(x.begin(),x.end(),v)-x.begin()) //xi>v  x is sorted
#define NUM(x,v) (POSU(x,v)-POSL(x,v))  //x is sorted
#define SORT(x) (sort(x.begin(),x.end())) // 0 2 2 3 4 5 8 9
#define REV(x) (reverse(x.begin(),x.end())) //reverse
#define TO(x,t,f) ((x)?(t):(f))
#define CLR(mat) memset(mat, 0, sizeof(mat))
#define FILV(x,a) fill(x.begin(),x.end(),a)
#define FILA(ar,N,a) fill(ar,ar+N,a)
#define NEXTP(x) next_permutation(x.begin(),x.end())
ll gcd(ll a,ll b){if(a<b)swap(a,b);if(a%b==0)return b;return gcd(b,a%b);}
ll lcm(ll a,ll b){ll c=gcd(a,b);return ((a/c)*(b/c)*c);}//saisyo kobaisu
#define pb push_back
#define pri(aa) cout<<(aa)<<endl
#define mp(x,y) make_pair(x,y)
#define fi first
#define se second
const ll INF=1e18+7;
const ll MOD=1e9+7;




int main(){
	FOR(i,1,10){
		ll M = 1;
		cout << (i%M) << endl;
	}
}*/

#include <bits/stdc++.h>
#define rep(i,a,b) for(int i = int(a); I < int(b); ++I)
using namespace std;
using ll = long long;
constexpr ll LONGINF=50000000000000000LL;
constexpr int INF=1050000000;
constexpr int MOD=998244353;



int main() {
	ll N;
	string S;
	cin >> N >> S;

	ll mod = 998244353;

	ll dp[(1<<10)][10]={};

	rep(i,1,N+1){
		ll dpnex[1<<10][10] = {};
		int c = S[i-1] - 'A';
		rep(j,0,(1<<10)){
			rep(k,0,10){
				( dpnex[j][k] += dp[j][k] ) %= mod;
				( dpnex[j | (1<<c)][c] += dp[j][c]) %= mod;
			}
		}
		rep(j,0,(1<<10)){
			rep(k,0,10){
				dp[j][k] = dpnex[j][k];
			}
		}
	}
	ll ans = 0;
	FOR(i,0,10){
		( ans += dp[(1<<10)-1][i] ) += mod;
	}
	cout << ans << endl;
}

