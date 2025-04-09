# ContraVis_next

基于流匹配（Flow Matching）的对比可视化系统，用于分析和比较深度学习模型在不同条件下的特征表示。

## 项目介绍

ContraVis_next 是一个用于分析深度学习模型特征表示差异的工具。它利用流匹配（Rectified Flow）技术来捕捉并可视化模型在原始数据和受干扰数据（如添加噪声、对抗样本）之间的特征变化。

本项目主要研究目标：
- 分析模型特征空间中的语义变化
- 对比正常样本和异常样本的特征表示
- 可视化特征变化过程，帮助理解模型决策边界

## 核心功能

- **特征变换**: 利用流匹配（Flow Matching）技术在不同特征表示之间建立平滑路径
- **语义比较**: 分析和比较原始数据与干扰数据（如添加随机噪声）的特征空间差异
- **可视化工具**: 使用降维技术（如UMAP）直观展示特征变化过程
- **异常检测**: 识别特征空间中的异常区域和语义变化

## 项目结构

```
ContraVis_next/
├── models/                  # 模型定义
│   ├── flow_matching.py     # 流匹配模型实现
│   ├── mlp.py               # MLP和TimeMLP模型
│   └── autoencoder.py       # 自编码器模型
├── data_generators/         # 数据生成工具
│   ├── backdoor_cifar.ipynb # CIFAR数据集后门攻击生成
│   └── sphere_manifold.ipynb# 球面流形数据生成（仅仅用于测试之前拓扑编辑的想法 已弃用）
├── dataset/                 # 数据集存储
├── model_weights/           # 训练模型权重 
├── results/                 # 实验结果和可视化
├── *.ipynb                  # 实验和测试笔记本
└── contraVisFlow.yml        # 环境配置文件
```

## 主要模型

### RectifiedFlow

项目的核心组件，实现了基于拓扑编辑距离的跨空间点匹配，基于流匹配（Flow Matching）算法实现双向transform，用于在特征空间中创建平滑路径。

主要方法：
- `linear_interpolate`: 在两个点之间线性插值
- `mse_loss`: 计算速度场的均方误差损失
- `euler_step_forward/backward`: 使用欧拉方法进行正向/反向步进

### TimeMLP

时间感知的多层感知机，用于预测流场中的向量场。

特点：
- 使用正弦位置编码（Sinusoidal Embedding）处理时间信息
- 支持多层隐藏层配置
- 适用于扩散模型和流匹配任务

## 使用方法

1. **环境配置**
   ```bash
   conda env create -f contraVisFlow.yml
   conda activate gnn_course
   ```

2. **数据生成**
   参考 `data_generators\backdoor_cifar.ipynb` 中的示例代码进行特征转换和比较

3. **alignedUmap基线**
   使用 `test_aligned_umap.ipynb`
4. **流匹配Transformation训练**
    使用 `test_laplacian_matching.ipynb`
5. **利用WL核进行匹配**
    使用 `test_topology_edit.ipynb`
## 依赖项

主要依赖库包括：
- PyTorch
- NumPy
- UMAP-learn
- Matplotlib
- 完整依赖见 `contraVisFlow.yml`
