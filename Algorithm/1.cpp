#include <eigen3/Eigen/Core>

#define N 100005
using namespace std;
class node
{
    int t, c;

public:
    node(int t, int c)
    {
        this->t = t;
        this->c = c;
    }
    node(node &b)
    {
        t = b.t;
        c = b.c;
    }
    node(const node &b)
    {
        t = b.t;
        c = b.c;
    }
    bool operator<(node &b)
    {
        return t < b.t;
    }
    bool operator<(const node &b)
    {
        return t < b.t;
    }
    void output()
    {
        cout << this->t;
    }
};
int main()
{
    priority_queue<node> q;
    node a(1, 1), b(2, 2);
    q.push(a);
}
