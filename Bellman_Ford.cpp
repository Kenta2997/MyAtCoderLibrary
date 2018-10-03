// http://sesenosannko.hatenablog.com/entry/2017/09/01/214852
// https://beta.atcoder.jp/contests/abc061/submissions/3322496
// http://judge.u-aizu.ac.jp/onlinejudge/review.jsp?rid=3166760#1
// 0-index
struct Bellman_Ford{
	struct edge{ll from, to, cost;};
	ll inf = 1e17;
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