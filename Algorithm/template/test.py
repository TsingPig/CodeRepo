def is_leap_year(year):
    # 判断是否为闰年
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def count_leap_years(m, n):
    # 确保 m < n
    if m > n:
        m, n = n, m

    # 计算闰年数量
    leap_years = [year for year in range(m, n + 1) if is_leap_year(year)]
    return leap_years


# 获取用户输入
m, n = map(int, input("请输入两个年份，用逗号分隔：").split(','))

# 计算并输出结果
leap_years = count_leap_years(m, n)
print(f"[{m}, {n}]之间一共有{len(leap_years)}个闰年")
