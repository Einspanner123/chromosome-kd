import torch

print('=== Tensor.size 和 Tensor.shape 的区别 ===')

# 创建一个示例张量
tensor = torch.randn(3, 4, 5)
print(f'张量形状: {tensor.shape}')
print(f'张量大小: {tensor.size()}')

print('\n1. 基本概念:')
print('   - tensor.size() 是一个方法（method）')
print('   - tensor.shape 是一个属性（attribute）')

print('\n2. 返回值类型:')
print(f'   tensor.shape 的类型: {type(tensor.shape)}')
print(f'   tensor.size() 的类型: {type(tensor.size())}')
print('   两者都返回 torch.Size 对象，它是 tuple 的子类')

print('\n3. 使用方式:')
print('   - tensor.shape 直接访问属性，不需要括号')
print('   - tensor.size() 需要调用方法，需要括号')


# 演示两种方式的使用
def demonstrate_usage():
    tensor = torch.randn(2, 3, 4)

    # 使用 shape 属性
    shape_attr = tensor.shape
    print(f'\n使用 shape 属性: {shape_attr}')

    # 使用 size() 方法
    size_method = tensor.size()
    print(f'使用 size() 方法: {size_method}')

    # 两者结果相同
    print(f'结果是否相同: {shape_attr == size_method}')


demonstrate_usage()

print('\n4. 特殊功能 - size() 方法可以带参数:')
tensor = torch.randn(3, 4, 5)

# size() 方法可以获取特定维度的大小
print('\n获取特定维度大小:')
print(f'   tensor.size(0): {tensor.size(0)}')  # 第0维大小
print(f'   tensor.size(1): {tensor.size(1)}')  # 第1维大小
print(f'   tensor.size(2): {tensor.size(2)}')  # 第2维大小

# shape 属性不能这样做，需要通过索引访问
print('\n使用 shape 属性获取特定维度:')
print(f'   tensor.shape[0]: {tensor.shape[0]}')
print(f'   tensor.shape[1]: {tensor.shape[1]}')
print(f'   tensor.shape[2]: {tensor.shape[2]}')

print('\n5. 性能差异:')
import time

tensor = torch.randn(100, 200, 300)

# 测试 shape 属性访问速度
start_time = time.time()
for _ in range(10000):
    s = tensor.shape
shape_time = time.time() - start_time

# 测试 size() 方法调用速度
start_time = time.time()
for _ in range(10000):
    s = tensor.size()
size_time = time.time() - start_time

print(f'   10000次 shape 属性访问耗时: {shape_time:.6f} 秒')
print(f'   10000次 size() 方法调用耗时: {size_time:.6f} 秒')
print('   shape 属性访问略微快一些，因为它不需要方法调用的开销')

print('\n6. 在您的代码中的应用:')
print('   在您的 bbox2roi 函数中:')
print('   - bboxes.size(0) 获取边界框的第一维大小（边界框数量）')
print('   - 这里必须使用 size() 方法，因为需要传入维度参数')
print('   - 等价的写法可以是 bboxes.shape[0]')


# 演示在您的代码中的等价用法
def demonstrate_equivalence():
    # 模拟边界框张量
    bboxes = torch.randn(5, 4)  # 5个边界框，每个4个坐标值

    # 您代码中的写法
    method_result = bboxes.size(0)

    # 等价的属性访问写法
    attr_result = bboxes.shape[0]

    print('\n等价性演示:')
    print(f'   bboxes.size(0): {method_result}')
    print(f'   bboxes.shape[0]: {attr_result}')
    print(f'   结果相同: {method_result == attr_result}')


demonstrate_equivalence()

print('\n7. 实际使用建议:')
suggestions = [
    '1. 如果只需要获取张量的整体形状，两者都可以使用，但 shape 属性更简洁',
    '2. 如果需要获取特定维度的大小，size(dim) 和 shape[dim] 等价',
    '3. 在性能敏感的代码中，优先使用 shape 属性',
    '4. 在您的具体代码中，bboxes.size(0) 可以替换为 bboxes.shape[0]',
]

for suggestion in suggestions:
    print(f'   {suggestion}')

print('\n8. 特殊情况:')
print('   当需要将形状信息用于某些方法时:')

tensor = torch.randn(2, 3, 4)

# size() 返回的 torch.Size 对象有一些额外的方法
size_obj = tensor.size()
print('   torch.Size 对象有额外方法，如:')
print(f'   - size_obj.numel(): {size_obj.numel()} (总元素数)')
print(f'   - len(size_obj): {len(size_obj)} (维度数)')

# 转换为普通元组
print(f'   转换为普通列表: {list(tensor.shape)}')

print('\n=== 总结 ===')
print('✓ tensor.shape 和 tensor.size() 在功能上基本等价')
print('✓ 主要区别在于访问方式：shape 是属性，size() 是方法')
print('✓ size() 可以接受维度参数，而 shape 需要通过索引访问特定维度')
print('✓ 在您的代码中，bboxes.size(0) 和 bboxes.shape[0] 完全等价')
print('✓ 推荐在不需要特定维度参数时使用 shape，在需要特定维度时可以选择任一方式')
