#include<iostream>
using namespace std;
using ll = long long;
int t;
int a[1000010];
void solve()
{
    int n; cin >> n;
    memset(a, 0, sizeof(a));
    ll s = 0; int mx = 0;
    for(int i = 0; i < n; i++) {
        cin >> a[i];
        if (a[i] > mx) mx = a[i];
        s += a[i];
    }
    s -= mx;
    if (s >= mx - 1) cout << "Yes" << endl;
    else cout << "No" << endl;
}
int main()
{
    cin >> t;
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    while(t--)solve();
    return 0;
}
