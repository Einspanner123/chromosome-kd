# 方向 J: SwiGLU 门控 FFN

> 创建日期: 2026-07-10
> 实验完成日期: 2026-07-12
> 状态: **❌ 已废弃**（无显著改进，代码已清理）
> 基线: +AdaLN-Zero (RF+Heun, mAP=0.856) + +DPM-Solver++ (mAP=0.863)
> 数据集: 24obj (24_chromosomes_object)
> 参数策略: B (D'=2/3·dim_feedforward, 参数量匹配)

---

## 1. 理论基础

### 1.1 SwiGLU 定义

SwiGLU (Swish-Gated Linear Unit) 出自 Shazeer 2020 "GLU Variants Improve Transformer"：

```
SwiGLU(x) = Swish(x · W_gate) ⊙ (x · W_value)
```

其中：
- `Swish(x) = x · σ(x)`，σ 为 sigmoid
- `W_gate` 和 `W_value` 是两个独立的线性变换（无偏置，跟随 LLaMA 惯例）
- ⊙ 为逐元素乘法

### 1.2 与当前激活函数对比

| 激活函数 | 公式 | 当前使用位置 |
|---|---|---|
| ReLU | `max(0, x)` | FFN / cls_head / reg_head / DynamicConv |
| SiLU (= Swish) | `x · σ(x)` | time_mlp / AdaLN-Zero 调制 MLP |
| **SwiGLU** | `Swish(xW_gate) ⊙ (xW_value)` | **本方案替换 FFN 中的 ReLU** |

### 1.3 为什么可能有效

1. **门控机制**：模型学到"哪些特征该通过"，比 ReLU 的硬截断更灵活
2. **平滑梯度**：Swish 的非单调性提供更丰富的梯度流，避免 ReLU 的死神经元问题
3. **大模型验证**：LLaMA/PaLM 等大模型中证明优于 ReLU/GELU

### 1.4 小模型上的不确定性

SwiGLU 的优势主要在大模型（7B+参数）中验证。本模型特征维度仅 256，FFN 中间维度 2048，属于小模型。效果不确定：
- 乐观：+0.001~0.005 mAP
- 中性：±0.001 mAP
- 悲观：-0.001~0.003 mAP（参数量变化导致优化不稳定）

---

## 2. 当前架构分析

### 2.1 FFN 结构

文件: `ldmdet/core/single_head.py`，类: `SingleDiffusionDetHead`

当前 FFN（lines 106-117）：

```python
self.linear1 = nn.Linear(feat_channels, dim_feedforward)    # 256 → 2048
self.dropout = nn.Dropout(dropout)
self.linear2 = nn.Linear(dim_feedforward, feat_channels)    # 2048 → 256
self.act = nn.ReLU(inplace=True)
```

前向传播（两条路径使用同一 FFN）：

**AdaLN-Zero 路径**（从 +AdaLN-Zero 起, line 297）：
```python
ffn_out = self.linear2(self.dropout(self.act(self.linear1(ffn_input))))
obj_flat = obj_flat + alpha2 * ffn_out
```

**Scale-Shift 路径**（DDPM baseline/RF+Heun, line 312）：
```python
obj_shortcut = self.linear2(self.dropout(self.act(self.linear1(obj_features))))
obj_features = obj_features + self.dropout3(obj_shortcut)
```

### 2.2 参数量

| 组件 | 参数量 |
|---|---|
| linear1 (256→2048) | 256×2048 + 2048 = 526,336 |
| linear2 (2048→256) | 2048×256 + 256 = 524,544 |
| **FFN 总计** | **1,050,880** |

### 2.3 推理成本

来自 `docs/research/inference_optimization/benchmark_report.json`：

| 组件 | 延迟 (ms) | 占单头比例 |
|---|---|---|
| RoIAlign | 3.53 | 52.2% |
| DynamicConv | 2.54 | 37.5% |
| Self-Attn | 0.15 | 2.2% |
| **FFN (linear1+linear2)** | **0.14** | **1.0%** |
| cls_head + reg_head | 0.15 | 2.2% |

**关键结论**：FFN 仅占单头前向 1.0%。SwiGLU 增加一个 GEMM，额外开销 ≈ 0.07ms/头 × 42 头 ≈ 2.9ms，相对总推理 335ms 仅 **+0.9%**。推理成本可忽略。

---

## 3. 替换方案设计

### 3.1 目标

仅替换 `SingleDiffusionDetHead` 的 FFN 激活函数（`linear1 → act → linear2` → SwiGLU FFN）。

**不修改的位置**：
- backbone (ResNet-50)：预训练权重绑定 ReLU
- FPN：预训练权重绑定 ReLU
- time_mlp / AdaLN-Zero 调制 MLP：已使用 SiLU
- cls_head / reg_head：结构为 `Linear → LayerNorm → ReLU`，非标准 FFN，不适合 SwiGLU
- DynamicConv：结构为动态卷积，不适合 SwiGLU

### 3.2 SwiGLU FFN 结构

```python
# SwiGLU FFN: 3 个线性层（无偏置，跟随 LLaMA 惯例）
w_gate:  feat_channels → swiglu_dim    # 256 → 1365
w_value: feat_channels → swiglu_dim    # 256 → 1365
w_out:   swiglu_dim → feat_channels    # 1365 → 256（有偏置）

# 前向: Swish(w_gate(x)) ⊙ w_value(x) → dropout → w_out
```

### 3.3 参数策略 B：D' = 2/3 · dim_feedforward

| 组件 | 参数量 |
|---|---|
| w_gate (256→1365, 无偏置) | 256×1365 = 349,440 |
| w_value (256→1365, 无偏置) | 256×1365 = 349,440 |
| w_out (1365→256, 有偏置) | 1365×256 + 256 = 349,696 |
| **SwiGLU FFN 总计** | **1,048,576** |
| vs 原 FFN | 1,050,880（差异 -0.2%，基本持平） |

`swiglu_dim = int(dim_feedforward * 2 / 3)` = `int(2048 * 2 / 3)` = `1365`

### 3.4 向后兼容设计

新增参数 `ffn_activation`，默认 `'relu'`：

```python
ffn_activation: str = 'relu'  # 'relu' 或 'swiglu'
```

当 `ffn_activation='relu'` 时，构建原有的 `linear1` / `linear2` / `act`，行为完全不变。
当 `ffn_activation='swiglu'` 时，构建 `w_gate` / `w_value` / `w_out`，不创建 `linear1` / `linear2` / `act`。

---

## 4. 详细代码修改规格

### 4.1 新增 SwiGLU 模块

文件: `ldmdet/core/single_head.py`
位置: `SingleDiffusionDetHead` 类之前（`NormalizedLinear` 类之后）

```python
class SwiGLU(nn.Module):
    """SwiGLU (Swish-Gated Linear Unit) FFN 模块.

    SwiGLU(x) = Swish(x · W_gate) ⊙ (x · W_value) → W_out

    参考: Shazeer 2020 "GLU Variants Improve Transformer"
    LLaMA 标准做法: gate/value 无偏置, out 有偏置, 中间维度 = 2/3 · dim_feedforward

    Args:
        in_features: 输入维度
        hidden_features: 中间维度 (推荐 2/3 · dim_feedforward)
        out_features: 输出维度 (通常 = in_features)
        dropout: dropout 概率
    """

    def __init__(self, in_features, hidden_features, out_features=None, dropout=0.0):
        super().__init__()
        out_features = out_features or in_features
        self.w_gate = nn.Linear(in_features, hidden_features, bias=False)
        self.w_value = nn.Linear(in_features, hidden_features, bias=False)
        self.w_out = nn.Linear(hidden_features, out_features, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = F.silu(self.w_gate(x))  # Swish = SiLU
        value = self.w_value(x)
        return self.w_out(self.dropout(gate * value))
```

### 4.2 SingleDiffusionDetHead.__init__ 修改

文件: `ldmdet/core/single_head.py`
位置: `SingleDiffusionDetHead.__init__` 方法

**新增参数**（在 `use_self_conditioning` 之后）：

```python
ffn_activation: str = 'relu',
```

**新增逻辑**（替换 lines 106-117 的 `linear1` / `dropout` / `linear2` / `act` 定义）：

```python
self.ffn_activation = ffn_activation

if ffn_activation == 'swiglu':
    swiglu_dim = int(dim_feedforward * 2 / 3)
    self.ffn = SwiGLU(
        in_features=feat_channels,
        hidden_features=swiglu_dim,
        out_features=feat_channels,
        dropout=dropout,
    )
    # 不创建 linear1 / linear2 / act
elif ffn_activation == 'relu':
    self.linear1 = nn.Linear(feat_channels, dim_feedforward)
    self.dropout = nn.Dropout(dropout)
    self.linear2 = nn.Linear(dim_feedforward, feat_channels)
    self.act = nn.ReLU(inplace=True)
else:
    raise ValueError(f"Unknown ffn_activation: {ffn_activation}")

# dropout1 / dropout2 / dropout3 / norm1 / norm2 / norm3 保持不变
# (无论 ffn_activation 是什么, 这些层都需要创建)
```

**关键**: `dropout1` / `dropout2` / `dropout3` / `norm1` / `norm2` / `norm3` 必须在 if/else 之外创建，因为它们在两条路径中都被使用。

### 4.3 前向传播修改

文件: `ldmdet/core/single_head.py`

**_forward_adaln_zero 方法** (line 297):

原代码:
```python
ffn_out = self.linear2(self.dropout(self.act(self.linear1(ffn_input))))
```

修改为:
```python
if self.ffn_activation == 'swiglu':
    ffn_out = self.ffn(ffn_input)
else:
    ffn_out = self.linear2(self.dropout(self.act(self.linear1(ffn_input))))
```

**_forward_scale_shift 方法** (line 312):

原代码:
```python
obj_shortcut = self.linear2(self.dropout(self.act(self.linear1(obj_features))))
```

修改为:
```python
if self.ffn_activation == 'swiglu':
    obj_shortcut = self.ffn(obj_features)
else:
    obj_shortcut = self.linear2(self.dropout(self.act(self.linear1(obj_features))))
```

### 4.4 不需要修改的文件

- `ldmdet/core/head.py`：`DiffusionDetHead` 不直接管理 FFN，通过 `single_head` 参数传递
- `ldmdet/core/dynamic_conv.py`：DynamicConv 不受影响
- 配置基链: `ldmdet_baseline.py` → `ldmdet_rf_heun_shifted_bs2.py` → `a2_rf_heun_adaln_24obj.py` / `a3_full_sota_24obj.py` / `a4_dpm_pp_24obj.py`：无需修改

---

## 5. 实验配置

### 5.1 +AdaLN-Zero-SwiGLU 配置

文件: `experiments/configs/ldmdet/directions/mainline_ablation_24obj/a2_swinglu_24obj.py`

```python
"""24obj SwiGLU 实验: +AdaLN-Zero + SwiGLU FFN (验证门控激活函数效果)

目的: 在 +AdaLN-Zero (RF+Heun) 基础上将 FFN 激活函数从 ReLU 替换为 SwiGLU
参数策略: B (D'=2/3·dim_feedforward=1365, 参数量匹配)
基线: +AdaLN-Zero (mAP=0.856)

SwanLab: 项目 'ldmdet-mainline-ablation-24obj', 实验 'a2_swinglu'
"""
_base_ = ['./a2_rf_heun_adaln_24obj.py']

model = dict(
    bbox_head=dict(
        single_head=dict(
            ffn_activation='swiglu',
        ),
    ),
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='ldmdet-mainline-ablation-24obj',
            experiment_name='a2_swinglu',
            description='24obj SwiGLU: +AdaLN-Zero + SwiGLU FFN (D=1365, 参数匹配) | bs=8, 150ep',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
```

### 5.2 +DPM-Solver++-SwiGLU 配置

文件: `experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_swinglu_24obj.py`

```python
"""24obj SwiGLU 实验: +DPM-Solver++ + SwiGLU FFN (验证 SwiGLU 与 DPM-Solver++ 叠加效果)

目的: 在 +DPM-Solver++ (SOTA) 基础上将 FFN 激活函数从 ReLU 替换为 SwiGLU
参数策略: B (D'=2/3·dim_feedforward=1365, 参数量匹配)
基线: +DPM-Solver++ (mAP=0.863)

SwanLab: 项目 'ldmdet-mainline-ablation-24obj', 实验 'a4_swinglu'
"""
_base_ = ['./a4_dpm_pp_24obj.py']

model = dict(
    bbox_head=dict(
        single_head=dict(
            ffn_activation='swiglu',
        ),
    ),
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(
        type='SwanlabVisBackend',
        init_kwargs=dict(
            project='ldmdet-mainline-ablation-24obj',
            experiment_name='a4_swinglu',
            description='24obj SwiGLU: +DPM-Solver++ + SwiGLU FFN (D=1365, 参数匹配) | bs=8, 150ep',
            api_key='Huzvq1fnDeqOwgQo2AMAI',
            resume='allow',
        ),
    ),
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
)
```

### 5.3 实验矩阵

| 实验 | 基线 | ffn_activation | D' | 预期 mAP | SwanLab experiment_name |
|---|---|---|---|---|---|
| +AdaLN-Zero-baseline | +AdaLN-Zero | relu | 2048 | 0.856 (已知) | a2_rf_heun_adaln |
| **+AdaLN-Zero-SwiGLU** | +AdaLN-Zero | swiglu | 1365 | 0.854-0.861 | a2_swinglu |
| +DPM-Solver++-baseline | +DPM-Solver++ | relu | 2048 | 0.863 (已知) | a4_dpm_pp |
| **+DPM-Solver++-SwiGLU** | +DPM-Solver++ | swiglu | 1365 | 0.860-0.868 | a4_swinglu |

---

## 6. 测试方案

### 6.1 单元测试规格

文件: `ldmdet/tests/test_core.py`
位置: `TestSingleDiffusionDetHead` 类中新增测试

**测试用例**:

1. **test_swiglu_module_basic**: SwiGLU 模块基础测试
   - 构造 SwiGLU(in_features=64, hidden_features=43, out_features=64)
   - 输入 `x = torch.randn(10, 64)`
   - 断言输出 shape == (10, 64)
   - 断言梯度可回传

2. **test_swiglu_module_param_count**: SwiGLU 参数量验证
   - 构造 SwiGLU(in_features=256, hidden_features=1365, out_features=256)
   - 统计参数量，断言 ≈ 1,048,576（与原 ReLU FFN 持平，容差 ±1%）

3. **test_single_head_swiglu_forward_adaln**: SwiGLU + AdaLN-Zero 前向
   - 构造 SingleDiffusionDetHead(feat_channels=64, dim_feedforward=128, time_conditioning='adaln_zero', ffn_activation='swiglu')
   - 断言输出 shape 与 relu 版本一致: cls_logits (2,10,24), pred_bboxes (2,10,4)

4. **test_single_head_swiglu_forward_scale_shift**: SwiGLU + Scale-Shift 前向
   - 构造 SingleDiffusionDetHead(feat_channels=64, dim_feedforward=128, time_conditioning='scale_shift', ffn_activation='swiglu')
   - 断言输出 shape 与 relu 版本一致

5. **test_single_head_swiglu_gradient_flow**: SwiGLU 梯度流
   - 构造 adaln_zero + swiglu 的 head
   - 前向 + 反向
   - 断言所有参数 (w_gate, w_value, w_out) 的 grad 非 None

6. **test_single_head_swiglu_deterministic**: SwiGLU 确定性
   - eval 模式下两次前向结果一致

7. **test_single_head_relu_backward_compat**: ReLU 向后兼容
   - 不传 ffn_activation 参数时，默认 'relu'，行为不变
   - 断言 hasattr(head, 'linear1') and hasattr(head, 'act')

8. **test_single_head_swiglu_no_legacy_attrs**: SwiGLU 模式不创建旧属性
   - ffn_activation='swiglu' 时，断言 not hasattr(head, 'linear1')
   - 断言 hasattr(head, 'ffn')

### 6.2 测试 fixture

```python
@pytest.fixture
def single_head_swiglu_adaln(self):
    return SingleDiffusionDetHead(
        num_classes=24,
        feat_channels=64,
        dim_feedforward=128,
        num_cls_convs=1,
        num_reg_convs=1,
        num_heads=4,
        pooler_resolution=7,
        dynamic_dim=32,
        dynamic_num=2,
        time_conditioning='adaln_zero',
        ffn_activation='swiglu',
    )

@pytest.fixture
def single_head_swiglu_scale_shift(self):
    return SingleDiffusionDetHead(
        num_classes=24,
        feat_channels=64,
        dim_feedforward=128,
        num_cls_convs=1,
        num_reg_convs=1,
        num_heads=4,
        pooler_resolution=7,
        dynamic_dim=32,
        dynamic_num=2,
        time_conditioning='scale_shift',
        ffn_activation='swiglu',
    )
```

### 6.3 运行命令

```bash
# 运行 SwiGLU 相关测试
cd /home/linkst/workspace/projects/chromosome-kd
source /home/linkst/data/miniconda3/etc/profile.d/conda.sh && conda activate chromo
python -m pytest ldmdet/tests/test_core.py -k "swiglu or SwiGLU" -v

# 运行全部 core 测试确保无回归
python -m pytest ldmdet/tests/test_core.py -v

# 启动训练
bash train.sh experiments/configs/ldmdet/directions/mainline_ablation_24obj/a2_swinglu_24obj.py
bash train.sh experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_swinglu_24obj.py
```

---

## 7. 风险与权衡

| 维度 | 评估 | 严重程度 |
|---|---|---|
| 推理成本 | +0.9%（2.9ms / 335ms），FFN 仅占 1% | 🟢 可忽略 |
| 参数量 | -0.2%（策略 B 参数匹配） | 🟢 无影响 |
| 小模型效果 | SwiGLU 优势主要在大模型验证 | 🟡 不确定 |
| 训练稳定性 | SwiGLU 门控初始化 | 🟢 w_out 零初始化可选 |
| 与 AdaLN 交互 | AdaLN 在 FFN 前调制，SwiGLU 在 FFN 内部门控，正交 | 🟢 无冲突 |
| 与 DPM-Solver++ 交互 | DPM-Solver++ 仅改推理采样，不改模型结构 | 🟢 无冲突 |

---

## 8. 红绿重构流程

### 8.1 RED: 编写测试（预期失败）

1. 在 `ldmdet/tests/test_core.py` 的 `TestSingleDiffusionDetHead` 类中新增 SwiGLU 测试
2. 导入 `SwiGLU` 类（预期 ImportError）
3. 测试 `ffn_activation='swiglu'` 参数（预期 TypeError）
4. 运行测试确认全部 RED

### 8.2 GREEN: 最小实现

1. 在 `ldmdet/core/single_head.py` 中新增 `SwiGLU` 类
2. 在 `SingleDiffusionDetHead.__init__` 中新增 `ffn_activation` 参数和分支逻辑
3. 修改 `_forward_adaln_zero` 和 `_forward_scale_shift` 的 FFN 前向
4. 运行测试确认全部 GREEN

### 8.3 REFACTOR: 重构和验证

1. 检查代码风格、命名一致性
2. 运行全部 `test_core.py` 确保无回归
3. 创建实验配置文件
4. 用 subagent（仅有本文档上下文）检查实现准确性

---

## 9. 成功标准

| 标准 | 阈值 |
|---|---|
| 单元测试全通过 | 8/8 SwiGLU 测试 + 全部原有测试无回归 |
| 参数量匹配 | SwiGLU FFN 参数量与 ReLU FFN 差异 < 1% |
| 前向输出 shape | 与 ReLU 版本完全一致 |
| 梯度流 | w_gate / w_value / w_out 梯度均可回传 |
| 向后兼容 | 不传 ffn_activation 时行为不变 |
| 推理速度 | SwiGLU 版本推理时间增加 < 2% |

---

## 10. 实验结果与结论

### 10.1 SwanLab 实验链接

| 实验 | SwanLab 链接 |
|---|---|
| +AdaLN-Zero-SwiGLU | https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/ooue3q2422mozagd3bjxx/chart |
| +DPM-Solver++-SwiGLU | https://swanlab.cn/@einspanner/ldmdet-mainline-ablation-24obj/runs/comza4feqdatcgbdus6gl/chart |

### 10.2 训练数据汇总

两个实验均于 2026-07-12 02:49 启动训练（150 epoch, batch_size=2, lr=5e-5）。

#### +AdaLN-Zero-SwiGLU（训练至 epoch 69 后手动终止）

| 指标 | +AdaLN-Zero-SwiGLU | +AdaLN-Zero-baseline (ReLU) | Δ |
|---|---|---|---|
| 最佳 mAP | **0.859** @ epoch 65 | **0.856** | +0.003 |
| 最佳 mAP50 | 0.990 | 0.990 | 0.000 |
| 最佳 mAP75 | 0.973 | 0.972 | +0.001 |
| 训练 epoch | 69 / 150 | 52 (记录) | — |

mAP 曲线（每 ~10 epoch 采样）:

| epoch | mAP | 趋势 |
|---|---|---|
| 6 | 0.769 | 早期收敛极快 |
| 11 | 0.802 | — |
| 16 | 0.833 | — |
| 26 | 0.838 | — |
| 36 | 0.837 | — |
| 42 | **0.853** | — |
| 65 | **0.859** | 最佳 |
| 69 | 0.859 | 持平 |

#### +DPM-Solver++-SwiGLU（训练至 epoch 66 后终止，连续 30 epoch 无改善）

| 指标 | +DPM-Solver++-SwiGLU | +DPM-Solver++-baseline (ReLU) | Δ |
|---|---|---|---|
| 最佳 mAP | **0.857** @ epoch 36 | **0.863** | **-0.006** |
| 最佳 mAP50 | 0.990 | 0.990 | 0.000 |
| 最佳 mAP75 | 0.970 | 0.974 | -0.004 |
| 训练 epoch | 66 / 150 | 117 (最佳) | — |

mAP 曲线（每 ~10 epoch 采样）:

| epoch | mAP | 趋势 |
|---|---|---|
| 6 | 0.772 | 早期收敛极快（+0.144 vs baseline） |
| 11 | 0.808 | — |
| 16 | 0.824 | — |
| 26 | 0.846 | — |
| 36 | **0.857** | **峰值，之后 30 epoch 无改善** |
| 41 | 0.847 | 下滑 |
| 44 | 0.852 | — |
| 66 | 0.843 | 继续下滑 |

日志关键信息：
```
the monitored metric did not improve in the last 30 records. best score: 0.857.
```

### 10.3 同 epoch 对比分析

#### +AdaLN-Zero-SwiGLU vs +AdaLN-Zero-baseline

| step | +AdaLN-Zero-baseline (ReLU) | +AdaLN-Zero-SwiGLU | Δ |
|---|---|---|---|
| 6 | 0.666 | 0.769 | +0.103 |
| 16 | 0.812 | 0.833 | +0.021 |
| 26 | 0.828 | 0.838 | +0.010 |
| 36 | 0.835 | 0.837 | +0.002 |
| 42 | 0.844 | 0.853 | +0.009 |
| 65 | — | **0.859** | — |
| **最佳** | **0.856** | **0.859** | **+0.003** |

#### +DPM-Solver++-SwiGLU vs +DPM-Solver++-baseline

| step | +DPM-Solver++-baseline (ReLU) | +DPM-Solver++-SwiGLU | Δ |
|---|---|---|---|
| 6 | 0.628 | 0.772 | +0.144 |
| 16 | 0.817 | 0.824 | +0.007 |
| 26 | 0.835 | 0.846 | +0.011 |
| 36 | 0.831 | **0.857** | +0.026 |
| 44 | 0.835 | 0.852 | +0.017 |
| 117 | **0.863** | — | — |
| **最佳** | **0.863** | **0.857** | **-0.006** |

### 10.4 结论

**SwiGLU 在 LDMDet 扩散检测模型中无显著改进，已废弃。**

核心发现：

1. **+AdaLN-Zero-SwiGLU +0.003 在噪声范围内**：24obj 耦合策略的 seed 方差为 ±0.001，epoch 间波动 ±0.005-0.010。+0.003 的差异不具统计显著性，需多种子实验才能确认，但收益太小不值得进一步验证。

2. **+DPM-Solver++-SwiGLU -0.006，明确负向**：早期峰值 0.857（step 36）是"虚假峰值"，之后连续 30 epoch 无改善并下滑到 0.843。SwiGLU + DPM-Solver++ 组合不如纯 ReLU FFN。**证伪了"SwiGLU 与 DPM-Solver++ 有正交互"的假说**。

3. **早期收敛加速是一致的**：两个实验在 step 6 都大幅领先（+AdaLN-Zero +0.103, +DPM-Solver++ +0.144），SwiGLU 的门控机制确实帮助模型更快学到有效特征。但这种早期优势**未能转化为最终性能提升**——随着训练进行，ReLU FFN 逐渐追平甚至反超。

4. **SwiGLU 的早期优势机制**：
   - Swish 的非单调性避免死神经元，训练初期保持梯度流
   - 门控信号在初期接近 0.5，起到类似 dropout 的正则化效果
   - 但随着训练进行，这种正则化变为容量瓶颈（D'=1365 < D=2048）

5. **SwiGLU 与 DPM-Solver++ 的交互**：之前假设"SwiGLU 的门控使 FFN 输出更一致，有利于 DPM-Solver++ 的多项式插值"被实验证伪。可能原因是 DPM-Solver++ 需要的是 $x_0^{pred}$ 在时间步间的一致性，而 SwiGLU 的门控可能在时间步间引入不一致的饱和模式。

### 10.5 代码清理记录

2026-07-12 完成代码清理：

- `ldmdet/core/single_head.py`：移除 SwiGLU 类、`ffn_activation` 参数、两条前向路径的 SwiGLU 分支
- `ldmdet/tests/test_core.py`：移除 TestSwiGLU 和 TestSingleDiffusionDetHeadSwiGLU 测试类
- `experiments/configs/ldmdet/directions/mainline_ablation_24obj/a2_swinglu_24obj.py`：已删除
- `experiments/configs/ldmdet/directions/mainline_ablation_24obj/a4_swinglu_24obj.py`：已删除
- `work_dirs/a2_swinglu_24obj/`：训练日志保留
- `work_dirs/a4_swinglu_24obj/`：训练日志保留
