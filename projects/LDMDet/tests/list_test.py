import torch

print("=== ROI特征提取代码优化分析 ===")

# 创建示例数据
bs, num_boxes = 2, 3
bboxes = torch.randn(bs, num_boxes, 4)
print(f"输入张量 bboxes 形状: {bboxes.shape}")
print(f"bboxes 内容:\n{bboxes}")

print("\n=== 当前实现分析 ===")
# 当前写法
proposal_boxes_current = list()
for b in range(bs):
    proposal_boxes_current.append(bboxes[b])
print(f"当前实现结果长度: {len(proposal_boxes_current)}")
for i, box in enumerate(proposal_boxes_current):
    print(f"  proposal_boxes[{i}] 形状: {box.shape}")

print("\n=== 优化方案 ===")

# 方案1: 使用列表推导式
proposal_boxes_v1 = [bboxes[b] for b in range(bs)]
print("方案1 - 列表推导式:")
print(f"  结果长度: {len(proposal_boxes_v1)}")
for i, box in enumerate(proposal_boxes_v1):
    print(f"  proposal_boxes[{i}] 形状: {box.shape}")

# 方案2: 直接使用unbind (推荐)
proposal_boxes_v2 = bboxes.unbind(0)  # 按第0维拆分
print("\n方案2 - 使用 unbind:")
print(f"  结果长度: {len(proposal_boxes_v2)}")
for i, box in enumerate(proposal_boxes_v2):
    print(f"  proposal_boxes[{i}] 形状: {box.shape}")

# 方案3: 使用 torch.chunk
proposal_boxes_v3 = torch.chunk(bboxes, bs, dim=0)
proposal_boxes_v3 = [chunk.squeeze(0) for chunk in proposal_boxes_v3]
print("\n方案3 - 使用 chunk:")
print(f"  结果长度: {len(proposal_boxes_v3)}")
for i, box in enumerate(proposal_boxes_v3):
    print(f"  proposal_boxes[{i}] 形状: {box.shape}")

# 方案4: 直接传递整个张量给bbox2roi (需要修改bbox2roi函数)
print("\n方案4 - 修改bbox2roi函数:")
print("  可以考虑修改bbox2roi函数直接接受批量张量，避免拆分操作")
print("  这需要重构bbox2roi函数，使其能够处理[b, n, 4]形状的输入")

print("\n=== 性能测试 ===")


def current_method(bboxes):
    bs = bboxes.shape[0]
    proposal_boxes = list()
    for b in range(bs):
        proposal_boxes.append(bboxes[b])
    return proposal_boxes


def optimized_method(bboxes):
    return list(bboxes.unbind(0))


# 创建更大的测试数据
large_bboxes = torch.randn(8, 100, 4)

import time


def benchmark(func, data, name, iterations=1000):
    # 预热
    for _ in range(10):
        _ = func(data)

    start = time.time()
    for _ in range(iterations):
        _ = func(data)
    end = time.time()

    return (end - start) / iterations * 1000  # 转换为毫秒


current_time = benchmark(current_method, large_bboxes, "current")
optimized_time = benchmark(optimized_method, large_bboxes, "optimized")

print(f"当前方法平均耗时: {current_time:.4f} ms")
print(f"优化方法平均耗时: {optimized_time:.4f} ms")

if current_time > optimized_time:
    speedup = current_time / optimized_time
    print(f"性能提升: {speedup:.2f}x")
else:
    speedup = optimized_time / current_time
    print(f"性能下降: {speedup:.2f}x")

print("\n=== 推荐的优化写法 ===")

print("# 推荐写法1: 使用 unbind (最简洁)")
print("proposal_boxes = list(bboxes.unbind(0))")

print("\n# 推荐写法2: 列表推导式 (更直观)")
print("proposal_boxes = [bboxes[b] for b in range(bs)]")

print("\n# 推荐写法3: 如果后续函数支持，直接使用整个张量")
print("# 这需要修改bbox2roi及相关函数")

print("\n=== 优化优势 ===")
advantages = [
    "1. 减少显式的循环操作，代码更简洁",
    "2. 使用PyTorch内置函数，可能有更好的性能",
    "3. 代码更符合Python和PyTorch的习惯用法",
    "4. 减少中间变量的创建",
    "5. 提高代码可读性",
]

for adv in advantages:
    print(adv)

print("\n=== 注意事项 ===")
notes = [
    "1. unbind 创建的是元组，需要转为列表以保持接口一致性",
    "2. 如果后续处理函数依赖于列表类型，需要保持类型一致",
    "3. 性能提升在大批量数据时更明显",
    "4. 在实际应用中，这部分通常不是性能瓶颈",
]

for note in notes:
    print(note)
