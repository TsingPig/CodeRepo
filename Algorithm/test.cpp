#include <iostream>
using namespace std;

int main()
{
    int n, q;
    cin >> n >> q;
    char a[n]; cin >> a;
    int num10 = 0, num01 = 0;
    for (int i = 0; i < q; i++)
    {
        int x, y;
        cin >> x >> y;
        for (int j = x + 1; j < y; j++)
        {
            if (a[j - 1] == '1' && a[j] == '0')
                num10++;
            if (a[j - 1] == '0' && a[j] == '1')
                num01++;
        }
        if (num10 == num01)
            cout << "YES" << endl;
        else
            cout << "NO" << endl;
    }
    return 0;
}
