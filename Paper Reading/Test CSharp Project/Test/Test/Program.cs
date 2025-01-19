using System;
using System.Collections.Generic;

class Program
{
    static void Main()
    {
        int[] numbers = { 5, 2, 9, 1 };

        Array.Sort(numbers, new IntComparer());
        Console.WriteLine(string.Join(", ", numbers)); // 1, 2, 5, 9
    }

    // 自定义比较器实现 IComparer<int>
    class IntComparer : IComparer<int>
    {
        public int Compare(int x, int y)
        {
            return x.CompareTo(y);
        }
    }
}
