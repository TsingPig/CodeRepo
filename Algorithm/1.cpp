#include <bits/stdc++.h>
using namespace std;
const int N = 15;
using ll = long long;

ll a[N], m[N], Mi[N];

// 扩展欧几里得算法
tuple<ll, ll, ll> exgcd(ll a, ll b)
{
    if (b == 0)
        return {1, 0, a};
    auto [x, y, d] = exgcd(b, a % b);
    return {y, x - a / b * y, d};
}

// 求 ax + by = c 的特解
pair<ll, ll> liEu(ll a, ll b, ll c)
{
    auto [x, y, d] = exgcd(a, b);
    if (c % d != 0)
        return {0, 0};
    a /= d;
    c /= d;
    b /= d;
    return {(x * c % b + b) % b, (y * c % a + a) % a};
}

ll inv(ll a, ll b)
{
    // ax mod b = 1
    // ax + 1y = b
    auto [x, y] = liEu(a, b, 1);
    return x;
}

int main()
{
    ll n, mul = 1; // mulΪ����m[i]�ĳ˻�
    cin >> n;
    for (int i = 1; i <= n; i++)
    {
        cin >> m[i] >> a[i];
        mul *= m[i];
    }

    ll X = 0;
    for (int i = 1; i <= n; i++)
    {
        Mi[i] = mul / m[i];
        ll x = 0, y = 0;
        ll cur = inv(Mi[i], m[i]);
        cur = (cur * Mi[i]) % mul;
        cur = (cur * a[i]) % mul;
        // X = (X + *Mi[i] * iv) % mul;
        X = (X + cur) % mul;
    }
    cout << X << endl;
    return 0;
}