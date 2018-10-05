// http://judge.u-aizu.ac.jp/onlinejudge/review.jsp?rid=2871291
class Seg_Tree{
public:
	vector<int> dat;
	int n;
	void initsize(int n0){
		int k=1;
		while(1){
			if(n0<=k){
				n=k;
				dat.resize(2*n-1);
				for(int i=0;i<2*n-1;i++)dat[i]=INT_MAX;
				break;
			}
			k*=2;
		}
	}

	//i banme wo x nisuru
	void update(int i,int x){
		i += n-1;
		dat[i] = x;
		while(i>0){
			i = (i-1) / 2;
			dat[i] = min(dat[i*2+1],dat[i*2+2]);
		}
	}

	//[a,b)
	int query0(int a,int b,int k,int l,int r){
		if(r<=a || b<=l)return INT_MAX;
		if(a<=l && r<=b)return dat[k];
		else{
			int vl = query0(a,b,k*2+1,l,(l+r)/2);
			int vr = query0(a,b,k*2+2,(l+r)/2,r);
			return min(vl,vr);
		}
	}

	//return min [a,b)
	int query(int a,int b){
		return query0(a,b,0,0,n);
	}
};