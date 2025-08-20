

# XAI-DiffNet: 面向EV充电需求预测的可信赖AI框架

---

### 论文题目: 多尺度可解释图学习在城市能源系统电动汽车充电需求预测中的应用

[GraphicalAbstract.pdf](https://github.com/user-attachments/files/21882811/GraphicalAbstract.pdf)

---

**XAI-DiffNet** 是一个端到端的深度学习框架，旨在为城市能源系统中的电动汽车（EV）充电需求提供**可信赖、准确且具备多尺度可解释性**的预测。该框架由两个核心且内在耦合的模块构成：

1. **预测模块**: 一个创新的 **双通道扩散图循环网络 (Dual-Channel Diffusion Graph Recurrent Network, DC-DGRN)**，通过捕捉复杂的时空依赖性来实现业界顶尖的预测精度。它运行在一个能模拟真实城市动态的复杂图结构之上。
2. **可解释性模块**: 一个 **多尺度解释生成器 (Multi-Scale Interpretation Generator)**，用于打开深度学习模型的「黑箱」。它提供层级化的解释，揭示模型做出特定预测的*原因*，从而为电网运营商和城市规划者建立信任并提供可操作的洞察。

该系统专为解决AI在关键基础设施应用中的「信任赤字」而设计，不仅关注预测准确性，更注重透明性、可验证性和面向决策的智能支持。

---

## 模型详情

### 1. 预测模块

预测引擎基于一个新型图学习架构，擅长建模城市充电网络中复杂的非欧几里得关系。

* **多语义融合小世界网络 (Multi-Semantic Fused Small-World Network)**: 我们突破了传统基于邻近度的简单图，构建了一个融合多层城市语义的图，包括：

  * 物理邻接性
  * 地理距离
  * 功能相似性（基于兴趣点 POI）
  * 出行流动性（基于OD流）
  * 历史需求相关性
* **双通道扩散图循环网络 (DC-DGRN)**: 一个先进的编码器-解码器架构，用于捕捉时空模式。其双通道设计允许同时建模并解耦不同类型的空间影响（如本地地理影响 vs. 远程语义影响）。

### 2. 多尺度解释生成器

可解释性模块通过提供层级化解释，使模型决策过程透明化，这对于建立运营商信任至关重要。

* **微观层面 (Microscopic Level)**: 识别出对预测结果影响最大的历史时间步和空间连接（邻近区域）。
* **中观层面 (Mesoscopic Level)**: 量化一个区域的预测在多大程度上依赖其局部邻里网络，揭示哪些区域具有稳健独立的模式，哪些区域高度耦合。
* **宏观层面 (Macroscopic Level)**: 评估每个区域对整体网络预测精度的系统性重要性，从而识别关键枢纽和潜在脆弱点。

---

## 项目结构

```
XAI-DiffNet/
├── main.py                     # 主脚本，运行整个工作流
├── config.py                   # 集中配置文件，管理所有参数
├── model.py                    # PyTorch模型定义 (DC-DGRN等)
├── data_loader.py              # 数据加载、预处理和划分
├── trainer.py                  # 模型训练与评估
├── interpreter.py              # 多尺度可解释性分析
├── visualization.py            # 绘制所有图表（地图、曲线等）
├── data/
│   ├── data_graph/             # 图数据
│   │   └── sw_fused_graph.csv  # 多语义融合图示例
│   ├── data_timeseries/
│   │   └── charging_demand.csv # 充电需求时间序列示例
│   └── data_geo/
│       └── shanghai_districts.shp # 地理空间数据（用于地图可视化）
├── outputs/
│   ├── checkpoints/            # 保存的模型权重 (如 best_model.pth)
│   ├── figures/                # 生成的图表和可视化结果
│   │   ├── microscopic/        # 微观解释地图
│   │   ├── mesoscopic/         # 中观依赖性地图
│   │   └── macroscopic/        # 宏观系统重要性地图
│   ├── logs/                   # 训练与分析日志
│   └── results/                # 包含性能指标和解释结果的CSV文件
└── requirements.txt            # 依赖包列表
```

---

## 环境要求

### 安装依赖

```bash
pip install -r requirements.txt
```

或手动安装核心依赖：

```bash
pip install torch numpy pandas scikit-learn matplotlib geopandas seaborn
```

---

## 使用说明

工作流通过 `main.py` 和 `config.py` 文件运行和配置。

### 1. 数据准备

1. 将数据文件放入 `data/` 文件夹下对应子目录。
2. 在 `config.py` 中更新路径指向您的数据集：

   * `ADJACENCY_MATRIX_FILE`: 融合图结构CSV文件
   * `SOC_DATA_FILE`: 充电需求时间序列CSV文件
   * `SHANGHAI_SHP_FILE`: shapefile文件（用于地图可视化）

### 2. 参数配置

在 `config.py` 中调整参数：

* **模型参数**: `SEQ_LEN`, `PRED_LEN`, `RNN_UNITS`, `NUM_RNN_LAYERS`
* **训练参数**: `NUM_EPOCHS_GNN`, `BATCH_SIZE`, `LEARNING_RATE_GNN`
* **可解释性参数**: `NUM_EPOCHS_MASK`, `LEARNING_RATE_MASK`

### 3. 运行工作流

执行主脚本：

```bash
python main.py
```

执行流程包括：

1. **数据加载与预处理**
2. **模型训练**（权重保存在 `outputs/checkpoints/`）
3. **模型评估**（保存性能指标）
4. **可解释性掩码训练**
5. **多尺度分析与可视化**（结果保存到 `outputs/`）

---

## 基线模型参考

1. 基于图神经网络（GNN）的预测模型

   * **参考链接:** [https://github.com/AIcharon-stt/Traffic-prediction-models-GNN](https://github.com/AIcharon-stt/Traffic-prediction-models-GNN)

---

## 输出结果

### 预测模块输出

* **性能指标**:

  * `outputs/results/metrics_original.csv`: 基础模型指标 (MAE, RMSE等)
  * `outputs/results/metrics_interpreted.csv`: 加入解释模块后的性能
* **损失曲线**:

  * `outputs/figures/gnn_loss_curve.png`: 训练与验证损失曲线

### 可解释性模块输出

* **掩码统计**:

  * `outputs/results/mask_statistics.csv`: 学习到的空间与时间掩码统计摘要
* **解释地图**:

  * `outputs/figures/microscopic/`: 区域影响力邻居图
  * `outputs/figures/mesoscopic/dependency_map.png`: 城市范围的局部依赖地图
  * `outputs/figures/macroscopic/systemic_importance_map.png`: 突出关键枢纽的系统性重要性地图

---

## 引用

如果您在研究中使用了 XAI-DiffNet 模型或相关思路，请引用我们的论文：

```
待补充
```

如有任何问题，请联系: \[[ttshi3514@163.com](mailto:ttshi3514@163.com)] 或 \[[1765309248@qq.com](mailto:1765309248@qq.com)]

---

