'''
【保研岛】蓝桥杯Python辅导课：课时1_Sort排序
'''

'''
sorted() 函数
Python内置的 sorted() 函数可以对任何可迭代对象进行排序，返回一个新的排序后的列表。
基本语法：
sorted(iterable, key=None, reverse=False)
iterable：需要排序的可迭代对象
key：排序规则，可以传入一个函数，指定排序依据（如按元素的某个属性排序）
reverse：是否反向排序，默认为 False，表示升序排序
'''
      
# 按照默认升序排序
nums = [5, 3, 8, 6, 7]
print(sorted(nums))  # 输出: [3, 5, 6, 7, 8]

# 降序排序
print(sorted(nums, reverse=True))  # 输出: [8, 7, 6, 5, 3]

# 按照字符串长度排序
words = ["apple", "banana", "kiwi", "cherry"]
print(sorted(words, key=len))  # 输出: ['kiwi', 'apple', 'cherry', 'banana']


'''
列表的 sort() 方法
sort() 是列表对象的一个方法，它会原地修改列表，直接排序，不返回新的列表。
list.sort(key=None, reverse=False)
key：排序规则，类似 sorted() 的 key
reverse：是否反向排序，默认为 False
'''
nums = [5, 3, 8, 6, 7]
nums.sort()  # 原地升序排序
print(nums)  # 输出: [3, 5, 6, 7, 8]

words = ["apple", "banana", "kiwi", "cherry"]
words.sort(key=len)  # 按照字符串长度排序
print(words)  # 输出: ['kiwi', 'apple', 'cherry', 'banana']


'''
使用 Lambda 表达式与排序
有时我们需要更加灵活的排序方式，可以使用 lambda 表达式作为 key 参数来指定排序规则。
'''
# 按照第二个字符排序
words = ["apple", "banana", "kiwi", "cherry"]
sorted_words = sorted(words, key=lambda x: x[1])
print(sorted_words)  # 输出: ['banana', 'cherry', 'kiwi', 'apple']
