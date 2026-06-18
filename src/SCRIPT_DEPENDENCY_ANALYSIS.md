# AsymFormer 项目 src 根目录脚本依赖关系分析

## 目录结构概览

```
src/
├── 核心模型定义
│   ├── AsymFormer.py              # 原始 AsymFormer 模型 (B0_T)
│   ├── AsymFormer_v1_5.py         # AsymFormer v1.5 (支持重参数化)
│   ├── MLPDecoder.py              # 解码器头
│   └── __init__.py                # 包初始化
│
├── 骨干网络架构
│   ├── convnext.py                # ConvNeXt 骨干 (含 asym_convnext_tiny)
│   ├── mix_transformer.py         # Mix Transformer (ViT 变体)
│   ├── mix_transformer_linear.py  # Mix Transformer (线性注意力版本)
│   ├── asym_unireplknet.py        # UniRepLKNet 非对称版本
│   └── rgbd_backbone_segformer_v2.py  # SegFormer V2 RGBD 骨干
│
├── 核心模块组件
│   ├── asym_block.py              # AsymBlock (ConvNeXt Block 的非对称版本)
│   ├── asymformer_block.py        # AsymFormerBlock V1 (dwconv/avgpool 版本)
│   ├── asymformer_block_v2.py     # AsymFormerBlock V2 (LIA 注意力版本)
│   ├── scc.py                     # SCC 模块 (跨模态注意力)
│   └── reparam_layers.py          # 重参数化层 (RepAsymConv2d 等)
│
├── 工具与配置
│   ├── training_utils.py          # 训练工具函数
│   ├── imagenet_utils.py          # ImageNet 工具
│   ├── dali_utils.py              # DALI 数据加载工具
│   ├── flops_counter.py           # FLOPs 计算工具
│   └── ema.py                     # 指数移动平均
│
├── 基准测试脚本
│   ├── benchmark_b2_b3_b4.py      # V11 系列模型基准测试
│   ├── benchmark_backbone_v2.py   # 骨干网络 V2 基准测试
│   ├── benchmark_bisenetv2.py     # BiSeNetV2 基准测试
│   ├── benchmark_lia.py           # LIA 模块基准测试
│   ├── benchmark_segformer.py     # SegFormer 基准测试
│   └── benchmark_backbone_v2.py   # 骨干网络 V2 基准测试
│
└── 其他工具脚本
    ├── test_asymformer_block_v2.py  # V2 模块测试
    ├── test_thop.py                 # Thop 库测试
    └── verify_flops_breakdown.py    # FLOPs 分解验证
```

---

## 核心模型依赖关系图

### 1. AsymFormer.py (原始版本)

**依赖关系:**
```
AsymFormer.py
├── mix_transformer.py (或 mix_transformer_linear.py)
│   ├── reparam_layers.py
│   └── timm
├── convnext.py
│   ├── reparam_layers.py
│   └── timm
├── MLPDecoder.py
├── scc.py
└── utils/flops_counter.py (用于 benchmark)
```

**主要组件:**
- `down_sample_block`: 下采样模块，整合 RGB 和 Depth 分支
- `B0_T`: 主模型类，使用 ConvNeXt-Tiny + Mix Transformer B0
- **特点**: 支持 `--test-resolution` 参数进行多分辨率基准测试

**使用场景:**
- 原始 AsymFormer 模型推理和训练
- 分辨率扩展性测试

---

### 2. AsymFormer_v1_5.py (重参数化版本)

**依赖关系:**
```
AsymFormer_v1_5.py
├── mix_transformer_linear.py
│   └── reparam_layers.py
├── convnext.py
│   └── reparam_layers.py
├── MLPDecoder.py
├── scc.py
└── reparam_layers.py (直接导入 switch_to_deploy)
```

**主要组件:**
- `down_sample_block`: 与 V1 类似，但支持重参数化
- `AsymFormer_v1_5`: 支持部署模式切换
- **关键方法**: `switch_to_deploy()` - 将模型转换为部署模式

**与 V1 的区别:**
- 使用 `mix_transformer_linear.py` 替代 `mix_transformer.py`
- 添加 `deploy` 参数控制部署模式
- 支持 `rgb_use_asym_dwconv` 和 `depth_use_asym_dwconv` 配置
- 增强的 `load_state_dict` 支持旧版 checkpoint 加载

---

### 3. MLPDecoder.py

**依赖关系:**
```
MLPDecoder.py
└── torch, torch.nn, torch.nn.functional
```

**主要组件:**
- `MLP`: 简单的 1x1 卷积投影层
- `DecoderHead`: 解码器头，融合多尺度特征
- **特点**: 支持 `switch_to_deploy()` 融合 BN 层

**被依赖:**
- AsymFormer.py
- AsymFormer_v1_5.py
- 所有使用解码器的模型

---

## 骨干网络依赖关系

### 4. convnext.py

**依赖关系:**
```
convnext.py
├── reparam_layers.py (RepAsymDWConv2d)
├── timm (DropPath, trunc_normal_, register_model)
└── asym_block.py (AsymBlock)
```

**主要组件:**
- `Block`: ConvNeXt Block，支持非对称 DWConv
- `ConvNeXt`: 原始 ConvNeXt 模型
- `AsymConvNeXt`: 非对称双分支 ConvNeXt
- `asym_convnext_tiny`: 工厂函数创建 Tiny 版本

**关键特性:**
- 支持 `use_asym_dwconv` 启用非对称 DWConv
- 支持 `deploy_dwconv` 部署模式
- `load_convnext_weights()`: 加载官方 ConvNeXt 预训练权重

---

### 5. mix_transformer.py / mix_transformer_linear.py

**依赖关系:**
```
mix_transformer.py
├── reparam_layers.py (RepAsymDWConv2d)
└── timm (DropPath, trunc_normal_)

mix_transformer_linear.py
└── (类似 mix_transformer.py，但使用线性注意力)
```

**主要组件:**
- `Mlp`: MLP 层，支持非对称 DWConv
- `Attention`: 注意力机制 (mix_transformer.py)
- `LinearAttention`: 线性注意力 (mix_transformer_linear.py)
- `OverlapPatchEmbed`: 重叠 patch 嵌入
- `mit_b0`: Mix Transformer B0 工厂函数

---

### 6. asym_unireplknet.py

**依赖关系:**
```
asym_unireplknet.py
├── pretrained_unireplknet.py
├── external/UniRepLKNet/unireplknet.py (动态导入)
└── torch, torch.nn
```

**主要组件:**
- `FusedSEFromTwo`: 融合两个 SE 模块
- `MergedUniRepLKNetBlock`: 合并的 UniRepLKNet Block
- `AsymUniRepLKNetBackbone`: 非对称 UniRepLKNet 骨干
- `asym_unireplknet_p/n/t`: 不同尺寸的工厂函数

**特点:**
- 需要外部 UniRepLKNet 代码库 (`external/UniRepLKNet/`)
- 支持加载 ImageNet-1K 预训练权重
- 支持重参数化部署

---

### 7. rgbd_backbone_segformer_v2.py

**依赖关系:**
```
rgbd_backbone_segformer_v2.py
└── asymformer_block_v2.py (AsymFormerBlockV2)
```

**主要组件:**
- `SegFormerRGBDPatchEmbedding`: RGBD Patch 嵌入层
- `SegFormerRGBDStageV2`: RGBD 阶段模块
- `SegFormerB2RGBDBackboneV2`: SegFormer B2 骨干 V2

**特点:**
- 使用 AsymFormerBlockV2 (带 LIA 注意力)
- 支持部分块使用 LIA (`lia_mask` 参数)
- 支持重参数化

---

## 核心模块组件依赖关系

### 8. asym_block.py

**依赖关系:**
```
asym_block.py
├── reparam_layers.py (RepAsymDWConv2d, RepAsymConfig)
└── timm (DropPath)
```

**主要组件:**
- `AsymBlock`: 非对称双分支 ConvNeXt Block
  - Main branch: 完整 ConvNeXt Block
  - Aux branch: 1/N 通道数
- **关键方法**: `reparameterize()` - 融合双分支为单分支

**设计模式:**
- 训练模式: 双输入 `(x_main, x_aux)`，双输出
- 部署模式: 单输入 `cat(x_main, x_aux)`，单输出

---

### 9. asymformer_block.py (V1 版本)

**依赖关系:**
```
asymformer_block.py
└── timm (DropPath，可选)
```

**主要组件:**
- `ChannelShuffle`: 通道混洗
- `PartialChannelShuffle`: 部分通道混洗
- `AsymSE`: 非对称 SE 模块
- `AsymFormerBlock`: AsymFormer Block V1
  - 支持 `dwconv` 和 `avgpool` 两种特征提取模式
  - 支持 `use_se` 启用 SE 注意力

**重参数化:**
- 融合 RepVGG-style DWConv 分支
- 融合 BN + MLP
- 合并分组卷积

---

### 10. asymformer_block_v2.py (V2 版本)

**依赖关系:**
```
asymformer_block_v2.py
└── timm (DropPath，可选)
```

**主要组件:**
- `ChannelShuffle`, `PartialChannelShuffle`: 通道混洗
- `AsymSEV2`: SE 模块 V2
- `LIAInteraction`: 局部交互注意力模块
- `AsymFormerBlockV2`: AsymFormer Block V2
  - 使用 LIA 注意力替代传统 DWConv
  - 支持 `use_se_lia` 控制是否启用 LIA

**架构改进:**
```
Stage 1: DW7 Mixing (Pre-Norm)
    ↓
Stage 2: AsymSE + LIA (Token 选择与交互)
    ↓
Stage 3: MLP (Pre-Norm, Groups=5)
```

---

### 11. scc.py

**依赖关系:**
```
scc.py
└── torch, torch.nn, torch.nn.functional
```

**主要组件:**
- `Cross_Atten_Lite_split`: 轻量级交叉注意力
- `SpatialAttention_max`: 空间注意力模块
- `SCC_Module`: SCC 融合模块
- **关键方法**: `switch_to_deploy()` - 融合 BN 层

**被依赖:**
- AsymFormer.py (down_sample_block)
- AsymFormer_v1_5.py (down_sample_block)

---

### 12. reparam_layers.py

**依赖关系:**
```
reparam_layers.py
└── torch, torch.nn, dataclasses
```

**主要组件:**
- `RepAsymConfig`: 重参数化配置
- `RepAsymConv2d`: 可重参数化非对称卷积
  - 分支: k×k, 1×k, k×1, identity
- `RepAsymDWConv2d`: 可重参数化非对称 DWConv

**被依赖:**
- convnext.py
- mix_transformer.py
- mix_transformer_linear.py
- asym_block.py
- AsymFormer_v1_5.py

---

## 基准测试脚本依赖关系

### 13. benchmark_b2_b3_b4.py

**依赖关系:**
```
benchmark_b2_b3_b4.py
├── AsymFormerV11B2.py (需要查看 archive 目录)
├── AsymFormerV11B2_v2.py
├── AsymFormer.py (B0_T)
└── utils/flops_counter.py (FlopsProfilerV2)
```

**功能:**
- 测试 V11 系列模型 (B2, B3, B4, B5)
- 测试 V2 版本骨干
- 对比原始 AsymFormer
- 输出: GFLOPs, Params, FP32/FP16 Latency & FPS

---

### 14. benchmark_backbone_v2.py

**依赖关系:**
```
benchmark_backbone_v2.py
├── rgbd_backbone_segformer_v2.py
└── utils/flops_counter.py
```

**功能:**
- 测试 SegFormerB2RGBDBackboneV2
- 输出: GFLOPs, Params, FP32/FP16 Latency & FPS

---

## 工具脚本

### 15. training_utils.py / imagenet_utils.py / dali_utils.py

**功能:**
- `training_utils.py`: 训练辅助函数
- `imagenet_utils.py`: ImageNet 数据集处理
- `dali_utils.py`: NVIDIA DALI 数据加载

---

### 16. flops_counter.py

**功能:**
- `FlopsProfilerV2`: 自定义 FLOPs 分析器
- 提供比 thop 更详细的模型分析

**被依赖:**
- AsymFormer.py
- benchmark_b2_b3_b4.py
- benchmark_backbone_v2.py

---

### 17. ema.py

**功能:**
- 指数移动平均 (Exponential Moving Average)
- 用于训练稳定化

---

## 依赖关系总结表

| 文件 | 主要依赖 | 被谁依赖 | 关键功能 |
|------|---------|---------|---------|
| **AsymFormer.py** | mix_transformer, convnext, MLPDecoder, scc | benchmark_b2_b3_b4 | 原始 AsymFormer 模型 |
| **AsymFormer_v1_5.py** | mix_transformer_linear, convnext, MLPDecoder, scc, reparam_layers | - | 重参数化版本 |
| **MLPDecoder.py** | torch | AsymFormer, AsymFormer_v1_5 | 解码器头 |
| **convnext.py** | reparam_layers, timm, asym_block | AsymFormer, AsymFormer_v1_5 | ConvNeXt 骨干 |
| **mix_transformer.py** | reparam_layers, timm | AsymFormer | Mix Transformer |
| **mix_transformer_linear.py** | reparam_layers, timm | AsymFormer_v1_5 | 线性注意力版本 |
| **asym_unireplknet.py** | pretrained_unireplknet, external/UniRepLKNet | - | UniRepLKNet 骨干 |
| **rgbd_backbone_segformer_v2.py** | asymformer_block_v2 | benchmark_backbone_v2 | SegFormer V2 骨干 |
| **asym_block.py** | reparam_layers, timm | convnext | 非对称 ConvNeXt Block |
| **asymformer_block.py** | timm (可选) | - | AsymFormer Block V1 |
| **asymformer_block_v2.py** | timm (可选) | rgbd_backbone_segformer_v2 | AsymFormer Block V2 (LIA) |
| **scc.py** | torch | AsymFormer, AsymFormer_v1_5 | SCC 融合模块 |
| **reparam_layers.py** | torch | convnext, mix_transformer, asym_block | 重参数化层 |
| **flops_counter.py** | torch, thop | AsymFormer, benchmark_* | FLOPs 分析 |

---

## 典型使用流程

### 训练流程

```python
# 1. 导入模型
from AsymFormer import B0_T
# 或
from AsymFormer_v1_5 import AsymFormer_v1_5

# 2. 创建模型
model = B0_T(num_classes=40, pretrained_backbone=True)

# 3. 训练
for batch in dataloader:
    rgb, depth, label = batch
    output = model(rgb, depth)
    loss = criterion(output, label)
    loss.backward()
    optimizer.step()
```

### 部署流程

```python
# 1. 创建模型 (deploy=True)
from AsymFormer_v1_5 import AsymFormer_v1_5
model = AsymFormer_v1_5(num_classes=40, deploy=True)

# 2. 加载权重
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint)

# 3. 推理
model.eval()
with torch.no_grad():
    # 部署模式接受拼接输入
    x = torch.cat([rgb, depth], dim=1)
    output = model(x)
```

### 重参数化流程

```python
# 1. 训练完成后切换
model = AsymFormer_v1_5(num_classes=40, deploy=False)
model.load_state_dict(checkpoint)

# 2. 切换到部署模式
model.switch_to_deploy()

# 3. 保存部署模型
torch.save(model.state_dict(), 'deploy_model.pth')
```

---

## 版本演进关系

```
AsymFormer (原始版)
    ↓
AsymFormer_v1_5 (添加重参数化支持)
    ↓
AsymFormerBlock V1 (dwconv/avgpool)
    ↓
AsymFormerBlock V2 (引入 LIA 注意力)
    ↓
SegFormer V2 Backbone (集成 V2 Block)
```

---

## 外部依赖

### Python 包依赖
```
torch
timm
thop (可选，用于 FLOPs 计算)
nvidia-dali (可选，用于数据加载)
```

### 外部代码库
```
external/UniRepLKNet/  # 用于 asym_unireplknet.py
```

---

## 总结

AsymFormer 项目的 src 根目录脚本形成了一个完整的 RGB-D 语义分割模型体系：

1. **核心模型**: AsymFormer.py 和 AsymFormer_v1_5.py 是主要入口
2. **骨干网络**: 支持 ConvNeXt、Mix Transformer、UniRepLKNet 多种骨干
3. **模块组件**: 提供 AsymBlock、AsymFormerBlock、SCC 等可复用模块
4. **重参数化**: 通过 reparam_layers.py 实现训练 - 部署转换
5. **基准测试**: 提供完整的性能评估脚本

依赖关系清晰，模块化设计良好，支持灵活配置和扩展。
