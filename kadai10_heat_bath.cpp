/*決まり

2次元 : f[y][x]

*/

#include <vector>
#include <iostream>
#include <cmath>
using namespace std;

// https://lipoyang.hatenablog.com/entry/2018/11/22/102006
// http://kobayashi.hub.hit-u.ac.jp/topics/rand.html
static unsigned int x_ = 123456789;
static unsigned int y_ = 362436069;
static unsigned int z_ = 521288629;
static unsigned int w_ = 88675123;
void xorshift128_seed(unsigned int seed){
	do{
		seed=seed*1812433253+1; seed^=seed<<13; seed^=seed>>17;
		x_ = 123464980 ^ seed;
		seed=seed*1812433253+1; seed^=seed<<13; seed^=seed>>17;
		y_ = 3447902351 ^ seed;
		seed=seed*1812433253+1; seed^=seed<<13; seed^=seed>>17;
		z_ = 2859490775 ^ seed;
		seed=seed*1812433253+1; seed^=seed<<13; seed^=seed>>17;
		w_ = 47621719 ^ seed;
	}while(x_==0&&y_==0&&z_==0&&w_==0);
}
unsigned int xorshift128(){
	unsigned int t;
	t = x_ ^ (x_ << 11);
	x_ = y_; y_ = z_; z_ = w_;
	w_ = (w_ ^ (w_ >> 19)) ^ (t ^ (t >> 8));
	return w_;
}
double xorshift(){
	return xorshift128()*2.32830643708e-10;
}


////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
// size
const int N = 50;

void random_spin(vector< vector<int> > &grid , int s = N){
	for(int i=0;i<s;i++){
		for(int j=0;j<s;j++){
			if(xorshift()<=0.500)grid[i][j] = -1;
			else grid[i][j] = 1;
		}
	}
}
void print_spin(const vector< vector<int> > &grid, int s = N){
	for(int i=0;i<s;i++){//y
		for(int j=0;j<s;j++){//x
			printf("%2d%c",grid[i][j],(j==s-1?'\n':' '));
		}
	}
}
int energy(const vector< vector<int> > &grid, int s = N){
	int res = 0;
	for(int i=0;i<s;i++){//y
		for(int j=0;j<s;j++){//x
			res += -grid[i][j]*grid[i][(j+1)%s];
			res += -grid[i][j]*grid[(i+1)%s][j];
		}
	}
	return res;
}
int magnetization(const vector< vector<int> > &grid, int s = N){
	int res=0;
	for(int i=0;i<s;i++){
		for(int j=0;j<s;j++){
			res += grid[i][j];
		}
	}
	return res;
}
int dE(const vector< vector<int> > &grid, int x, int y,int s = N){
	// (x,y)のスピンを変えた時の変化 E -> E+dE
	int res = 0;
	res += grid[y][x]*grid[y][(x-1+s)%s]*2;
	res += grid[y][x]*grid[y][(x+1)%s]*2;
	res += grid[y][x]*grid[(y-1+s)%s][x]*2;
	res += grid[y][x]*grid[(y+1)%s][x]*2;
	return res;
}
void printg(const vector< vector<int> > &grid,int s = N){
	for(int i=0;i<s;i++){
		for(int j=0;j<s;j++){
			printf("%d %d %d\n",j,i,grid[i][j]);
		}
	}
}

int main(){
	// grid[y][x]
	xorshift128_seed(1);
	vector< vector<int> > grid(N,vector<int>(N));
	random_spin(grid);
	//print_spin(grid);
	int E = energy(grid);
	//printf("E = %d\n",E);
	//cout << 0 << " " << E << endl;
	for(int t=0;t<2000;t++){//tMCS
		for(int i=0;i<N;i++){
			for(int j=0;j<N;j++){
				if((i+j)&1==t&1)continue;
				int de = dE(grid,j,i);
				double p = (double)1.0/(1.0+exp((double)de/6.0));
				if(xorshift()<=p){
					grid[i][j] *= -1;
					E += de;
				}
			}
		}
		//cout << t+1 << " " << E << endl;
	}
	//print_spin(grid);
	//printf("E = %d\n",energy(grid));
	//printf("M = %d\n",magnetization(grid));
	printg(grid);

}
