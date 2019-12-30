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
