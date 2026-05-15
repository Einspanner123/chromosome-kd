import time

import torch

print('=== 修正后的性能比较 ===')

# 设置测试数据
batch_size = 1000
half_dim = 256

timesteps = torch.randn(batch_size)
freqs = torch.randn(half_dim)

print(f'时间步形状: {timesteps.shape}')
print(f'频率参数形状: {freqs.shape}')


# 方法1: 元素级乘法（广播乘法）
def element_wise_multiplication():
    args = timesteps[:, None] * freqs[None, :]
    return args


# 方法2: 使用矩阵乘法 (@)
def matrix_multiplication():
    args = timesteps[:, None] @ freqs[None, :]
    return args


# 方法3: 使用 torch.matmul
def torch_matmul_method():
    args = torch.matmul(timesteps[:, None], freqs[None, :])
    return args


# 方法4: 使用 torch.outer
def torch_outer_method():
    args = torch.outer(timesteps, freqs)
    return args


# 验证四种方法结果一致
result1 = element_wise_multiplication()
result2 = matrix_multiplication()
result3 = torch_matmul_method()
result4 = torch_outer_method()

print('\n=== 结果验证 ===')
print(f'结果形状: {result1.shape}')
print(f'元素级 vs @: {torch.allclose(result1, result2)}')
print(f'元素级 vs matmul: {torch.allclose(result1, result3)}')
print(f'元素级 vs outer: {torch.allclose(result1, result4)}')

print('\n=== 修正后的性能测试 ===')


# 修正后的性能测试函数
def time_function_corrected(func, name, warmup=100, iterations=1000):
    # 预热
    for _ in range(warmup):
        func()

    # 清空CUDA缓存（如果使用GPU）
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # 实际测试
    start_time = time.perf_counter()  # 使用更精确的计时器
    for _ in range(iterations):
        func()

    if torch.cuda.is_available():
        torch.cuda.synchronize()  # 确保所有CUDA操作完成

    end_time = time.perf_counter()

    avg_time = (end_time - start_time) / iterations * 1000  # 转换为毫秒
    return avg_time


# 进行多次测试取平均值
def benchmark_multiple_runs(func, name, runs=5):
    times = []
    for run in range(runs):
        avg_time = time_function_corrected(func, name)
        times.append(avg_time)

    mean_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(
        f'{name}: {mean_time:.4f} ms (min: {min_time:.4f}, max: {max_time:.4f})'
    )
    return mean_time


# 运行基准测试
print('进行基准测试...')
methods = [
    (element_wise_multiplication, '元素级乘法 (* )'),
    (matrix_multiplication, '矩阵乘法 (@)'),
    (torch_matmul_method, 'torch.matmul'),
    (torch_outer_method, 'torch.outer'),
]

results = []
for func, name in methods:
    avg_time = benchmark_multiple_runs(func, name)
    results.append((name, avg_time))

# 排序结果
results.sort(key=lambda x: x[1])
print('\n=== 性能排名 ===')
for i, (name, elapsed) in enumerate(results, 1):
    print(f'{i}. {name}: {elapsed:.4f} ms')

print('\n=== 性能分析 ===')
fastest = results[0][1]
for name, elapsed in results:
    relative = elapsed / fastest
    print(f'{name} 比最快的方法慢 {relative:.2f} 倍')
