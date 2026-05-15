import time

import torch

print('=== bbox2roi 函数优化分析 ===')


def current_bbox2roi(bbox_list):
    """当前实现"""
    rois_list = []
    for id, bboxes in enumerate(bbox_list):
        img_inds = bboxes.new_full((bboxes.shape[0], 1), id)
        rois = torch.cat([img_inds, bboxes], dim=-1)
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois


def official_bbox2roi(bbox_list):
    """官方实现（稍作修改）"""
    from mmdet.structures.bbox import get_box_tensor

    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        bboxes = get_box_tensor(bboxes)  # 处理 BaseBoxes 类型
        img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
        rois = torch.cat([img_inds, bboxes], dim=-1)
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois


print('当前实现特点:')
print('1. 使用 for 循环遍历 bbox_list')
print('2. 为每个图像创建索引列')
print('3. 使用 torch.cat 连接索引和边界框')
print('4. 最后再次使用 torch.cat 合并所有 ROIs')

print('\n=== 性能测试 ===')


# 创建测试数据
def create_test_data(num_images=5, boxes_per_image=100):
    bbox_list = []
    for i in range(num_images):
        # 模拟边界框数据 (x1, y1, x2, y2)
        bboxes = torch.randn(boxes_per_image, 4)
        bbox_list.append(bboxes)
    return bbox_list


# 测试不同规模的数据
test_configs = [
    (1, 10),  # 1张图像，每张10个框
    (5, 100),  # 5张图像，每张100个框
    (10, 200),  # 10张图像，每张200个框
]


def benchmark_function(func, bbox_list, name, iterations=100):
    # 预热
    for _ in range(10):
        _ = func(bbox_list)

    # 实际测试
    start_time = time.time()
    for _ in range(iterations):
        _ = func(bbox_list)
    end_time = time.time()

    avg_time = (end_time - start_time) / iterations * 1000  # 转换为毫秒
    return avg_time


for num_images, boxes_per_image in test_configs:
    print(f'\n测试配置: {num_images} 张图像，每张 {boxes_per_image} 个边界框')
    test_data = create_test_data(num_images, boxes_per_image)

    current_time = benchmark_function(
        current_bbox2roi, test_data, 'current', 50
    )
    print(f'  当前实现平均耗时: {current_time:.4f} ms')

print('\n=== 可能的优化方案 ===')


def optimized_bbox2roi_v1(bbox_list):
    """优化版本1: 减少中间列表"""
    rois_parts = []
    for img_id, bboxes in enumerate(bbox_list):
        img_inds = bboxes.new_full((bboxes.shape[0], 1), img_id)
        rois_parts.append(img_inds)
        rois_parts.append(bboxes)

    # 交替连接索引和边界框，然后重塑
    all_parts = torch.cat(rois_parts, dim=0)
    # 重新排列为正确的格式
    # 这种方法实际上更复杂，不推荐


def optimized_bbox2roi_v2(bbox_list):
    """优化版本2: 预分配张量"""
    # 计算总的边界框数量
    total_boxes = sum(b.shape[0] for b in bbox_list)
    if len(bbox_list) == 0:
        return torch.empty(0, 5)

    # 假设边界框是4维的
    box_dim = bbox_list[0].shape[1]
    rois = torch.empty(
        total_boxes,
        box_dim + 1,
        dtype=bbox_list[0].dtype,
        device=bbox_list[0].device,
    )

    start_idx = 0
    for img_id, bboxes in enumerate(bbox_list):
        num_boxes = bboxes.shape[0]
        end_idx = start_idx + num_boxes
        rois[start_idx:end_idx, 0] = img_id  # 设置图像索引
        rois[start_idx:end_idx, 1:] = bboxes  # 设置边界框坐标
        start_idx = end_idx

    return rois


def optimized_bbox2roi_v3(bbox_list):
    """优化版本3: 使用 torch.stack 当所有图像有相同数量的边界框时"""
    # 这只适用于特殊情况，一般不适用
    pass


print('优化方案分析:')

print('1. 预分配张量方法 (optimized_bbox2roi_v2):')
print('   - 优势: 避免多次内存分配和连接操作')
print('   - 劣势: 需要预先计算总大小')
print('   - 适用场景: 边界框数量较大的情况')

print('\n2. 减少 torch.cat 调用:')
print('   - 当前实现使用了 O(n) 次 cat 操作')
print('   - 优化后只使用 1 次赋值操作')

print('\n=== 实际性能对比 ===')

# 测试优化版本
test_data = create_test_data(5, 200)

current_time = benchmark_function(current_bbox2roi, test_data, 'current', 50)
optimized_time = benchmark_function(
    optimized_bbox2roi_v2, test_data, 'optimized', 50
)

print(f'当前实现平均耗时: {current_time:.4f} ms')
print(f'优化实现平均耗时: {optimized_time:.4f} ms')

if current_time > optimized_time:
    improvement = (current_time - optimized_time) / current_time * 100
    print(f'性能提升: {improvement:.2f}%')
else:
    improvement = (optimized_time - current_time) / optimized_time * 100
    print(f'性能下降: {improvement:.2f}%')

print('\n=== 实际优化实现 ===')


def final_optimized_bbox2roi(bbox_list):
    """
    优化后的 bbox2roi 实现
    """
    # 处理空列表的情况
    if len(bbox_list) == 0:
        return torch.empty(0, 5)  # 假设边界框是4维

    # 计算总的边界框数量
    total_boxes = sum(b.shape[0] for b in bbox_list)

    # 获取边界框维度（假设所有边界框维度相同）
    box_dim = bbox_list[0].shape[1]

    # 预分配结果张量
    rois = torch.empty(
        total_boxes,
        box_dim + 1,
        dtype=bbox_list[0].dtype,
        device=bbox_list[0].device,
    )

    # 填充数据
    start_idx = 0
    for img_id, bboxes in enumerate(bbox_list):
        num_boxes = bboxes.shape[0]
        end_idx = start_idx + num_boxes

        # 设置图像索引
        rois[start_idx:end_idx, 0] = img_id

        # 设置边界框坐标
        rois[start_idx:end_idx, 1:] = bboxes

        start_idx = end_idx

    return rois


print('优化实现特点:')
print('1. 预先分配内存，避免多次重新分配')
print('2. 直接赋值而非连接操作')
print('3. 处理边界情况（空列表）')
print('4. 保留原始数据类型和设备')

print('\n=== 与官方实现的比较 ===')

comparison_points = {
    '兼容性': {
        '当前实现': '仅支持 Tensor 类型输入',
        '官方实现': '支持 Tensor 和 BaseBoxes 类型输入',
        '优化实现': '需要添加 BaseBoxes 支持',
    },
    '性能': {
        '当前实现': '多次连接操作，内存效率较低',
        '官方实现': '与当前实现类似',
        '优化实现': '预分配内存，更高内存效率',
    },
    '可读性': {
        '当前实现': '逻辑清晰，易于理解',
        '官方实现': '同样清晰，增加了类型处理',
        '优化实现': '稍复杂，但注释可以改善',
    },
}

for category, implementations in comparison_points.items():
    print(f'\n{category}:')
    for impl, desc in implementations.items():
        print(f'  {impl}: {desc}')

print('\n=== 建议 ===')

recommendations = [
    '1. 如果追求极致性能，可以采用预分配内存的优化实现',
    '2. 如果注重代码可读性和兼容性，当前实现已经足够好',
    '3. 可以考虑添加对 BaseBoxes 类型的支持以提高兼容性',
    '4. 在实际使用中，bbox2roi 通常不是性能瓶颈，优化收益有限',
    '5. 如果处理大量边界框（数千个），优化版本会有明显改善',
]

for rec in recommendations:
    print(rec)
