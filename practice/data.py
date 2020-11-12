from math import floor


class treeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


def preorder(root):
    if root is None:
        return []
    else:
        return [root.val] + preorder(root.left) + preorder(root.right)


def level_order(root):
    if not root:
        return []
    res = []
    nodequeue = [root]
    while nodequeue:
        root = nodequeue.pop(0)
        res.append(root.val)
        if root.left:
            nodequeue.append(root.left)
        if root.right:
            nodequeue.append(root.right)
    return res


class MaxHeap:
    def __init__(self):
        self.data = [None]

    def get_size(self):
        pass

    def insert(self, value):
        self.data.append(value)
        index = self.get_size
        while index > 1 and self.data[index] > self.data[index // 2]:
            self.data[index], self.data[index // 2] = self.data[index // 2], self.data[index]
            index = index // 2
        pass


def sift_down(self, index):
    while 2 * index <= self.get_size():
        # 左子结点的索引
        child = 2 * index
        # 如果右子结点存在且比左子结点大，则应与右子结点交换
        if 2 * index + 1 <= self.get_size() and self.data[2 * index + 1] > self.data[2 * index]:
            child += 1  # 右子结点的索引
        # 如果当前结点的值小于子结点中的较大者，则应继续向下交换，否则结束
        if self.data[index] < self.data[child]:
            self.data[index], self.data[child] = self.data[child], self.data[index]
            index = child
        else:
            break


# 删除堆顶元素（最大值）
def remove(self):
    if self.is_empty():
        raise ("Unable to remove from an empty heap.")
    # 用堆的最后一个元素替代堆顶元素，然后删除最后一个元素
    self.data[1], self.data[self.get_size()] = self.data[self.get_size()], self.data[1]
    self.data.pop()
    # 从堆顶向下调整
    self.sift_down(1)


i = 100
parent = i // 2
left = 2 * i
right = 2 * i + 1

a = [3, 5, 7]

print(len(a))
print()

print(3 // 2)
# all_num = 0
# def minus (num):#初始输入10
#     global all_num
#     if num == 0:
#         all_num = all_num +1
#         return 0
#     if num == -1:
#         return 0
#     if num > 0:
#         minus (num-1)
#         minus (num-2)
#
# num = 10
# minus(num)
# print (all_num)


