#include <bits/stdc++.h>
using namespace std;
using ll = long long;


int main()
{
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    int a[] = {4, 1, 2, 4};

    multiset<int> s(a, a + 3);
    cout << *s.begin() << endl;
    cout << *s.rbegin() << endl;
    return 0;
}