/// https://atcoder.jp/contests/abc036/submissions/6785659

#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

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
  
int main(){
	int N;
	cin >> N;
	vector<int> a(N);
	for(auto &x:a){
		cin >> x;
	}
	zatsu<int> z(a);
	for(auto x:a){
		cout << z.za[x] << endl;
	}
}