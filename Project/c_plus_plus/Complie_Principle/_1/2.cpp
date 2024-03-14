#include<bits/stdc++.h>
using namespace std;
int g[10][10];
int main(){
    memset(g,1,sizeof(g));
    for(int i=0;i<10;i++){
        for(int j=0;j<10;j++){
            cout<<g[i][j] << " ";
        }cout << endl;
        cout<<endl;
    }
    return 0;
}