#include <iostream>
using namespace std;

class A
{
    int *p;

public:
    A(int x = 0)
    {
        p = new int;
        *p = x;
    }
    A(A &obj)
    {
        p = new int;
        *p = (*obj.p) + 1;
    }
    A operator+(int x)
    {
        A ans;
        *ans.p = *p + x;
        cout << *ans.p << endl;
        return ans;
    }
    A& operator=(const A &obj)
    {
        if (this == &obj)
            return *this;
        *p = *obj.p + 2;
        cout << *p << endl;
        return *this;
    }
    virtual ~A()
    {
        cout << -(*p) << endl;
        delete p;
    }
};
int main()
{
    A a1(1), a2;
    a2 = a1 + 3;
    return 0;
}
