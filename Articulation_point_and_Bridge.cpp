//無向グラフ
// http://kagamiz.hatenablog.com/entry/2013/10/05/005213
// http://judge.u-aizu.ac.jp/onlinejudge/review.jsp?rid=3424144
// http://judge.u-aizu.ac.jp/onlinejudge/review.jsp?rid=3424140
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