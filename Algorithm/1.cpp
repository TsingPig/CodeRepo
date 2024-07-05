#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// 1 7 5 6 4 
// 1 6 5 7 4
void quickSort(int* a, int n)
{
    int pos = n / 2; int x = a[pos];
    int i = pos - 1, j = pos + 1;
    while(i > 0 && j < n) {
        if(a[i] < x) {
        }
    }
}

int main()
{
    int a[10];
    for(int i = 0; i < 10; i++)
        scanf("%d", &a[i]);
    quickSort(a, 10);
    for(int i = 0; i < 10; i++)
        printf("%d ", a[i]);
    return 0;
}