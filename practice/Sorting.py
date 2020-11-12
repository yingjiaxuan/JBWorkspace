import random


def bubble_sort(lst):
    if not lst:
        return []
    else:
        for i in range(len(lst)):
            for j in range(1, len(lst) - i):
                if lst[j - 1] > lst[j]:
                    lst[j - 1], lst[j] = lst[j], lst[j - 1]
        return lst


def select_sort(lst):
    if not lst:
        return []
    else:
        for i in range(len(lst) - 1):
            smallest = i
            for j in range(i, len(lst)):
                if lst[j] < lst[smallest]:
                    smallest = j
                lst[i], lst[smallest] = lst[smallest], lst[i]
        return lst


def insert_sort(lst):
    if not lst:
        return []
    else:
        for i in range(len(lst)):
            j = i
            while j > 0 and lst[j] < lst[j - 1]:
                lst[j - 1], lst[j] = lst[j], lst[j - 1]
                j = j - 1
        return lst


def shell_sort(lst):
    if not lst:
        return []
    h = 1
    while h < len(lst) / 3:
        h = 3 * h + 1
    while h >= 1:
        for i in range(int(h), len(lst)):
            j = 1
            while j >= h and lst[j] < lst[j - h]:
                lst[j], lst[j - h] = lst[j - h], lst[j]
                j = j - 1
        h = h / 3
    return lst


def mergesort(seq):
    """归并排序"""
    if len(seq) <= 1:
        return seq
    mid = len(seq) // 2  # 将列表分成更小的两个列表
    # 分别对左右两个列表进行处理，分别返回两个排序好的列表
    left = mergesort(seq[:mid])
    right = mergesort(seq[mid:])
    # 对排序好的两个列表合并，产生一个新的排序好的列表
    return merge(left, right)


def merge(left, right):
    """合并两个已排序好的列表，产生一个新的已排序好的列表"""
    result = []  # 新的已排序好的列表
    i = 0  # 下标
    j = 0
    # 对两个列表中的元素 两两对比。
    # 将最小的元素，放到result中，并对当前列表下标加1
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result += left[i:]
    result += right[j:]
    return result


def quick_sort(lst):
    if not lst:
        return []
    random.shuffle(lst)
    base = lst[0]
    left = quick_sort([x for x in lst[1:] if x <= base])
    right = quick_sort([x for x in lst[1:] if x > base])
    return left + [base] + right


list_0 = []
list_1 = [22, 5, 37, 1, 63, 24]
# print(bubble_sort(list_0))
# print(bubble_sort(list_1))
# print(insert_sort(list_1))
# print(shell_sort(list_1))
# print(mergesort(list_1))
# print(quick_sort(list_1))

print((233) % 8 ^ 5)
