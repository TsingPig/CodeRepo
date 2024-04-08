#include <bits/stdc++.h>
using namespace std;
using ll = long long;

#define lowbit(x) (x & -x)
const int N = 5e5 + 10;
int n;
int a[N];
int tr[N];

int query(int idx)
{
    int ans = 0;
    while (idx > 0)
    {
        ans += tr[idx];
        idx -= lowbit(idx);
    }
    return ans;
}

void update(int idx, int val)
{
    while (idx <= n)
    {
        tr[idx] += val;
        idx += lowbit(idx);
    }
}

ll solve()
{
    unordered_set<int> s(a, a + n);
    vector<int> b(s.begin(), s.end());
    sort(b.begin(), b.end());

    unordered_map<int, int> d;
    for (int i = 0; i < b.size(); i++)
        d[b[i]] = i + 1;
    for (int i = 0; i < n; i++)
        a[i] = d[a[i]];
    ll res = 0;
    for (int i = 0; i < n; i++)
    {
        int x = a[i];
        int q = query(x);
        res += i - q;
        update(x, 1);
    }
    return res;
}

int main()
{
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    cin >> n;
    for (int i = 0; i < n; i++)
        cin >> a[i];
    memset(tr, 0, sizeof(tr));
    cout << solve() << endl;
    return 0;
}
