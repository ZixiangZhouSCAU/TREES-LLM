# TREES-LLM 代码库

> TREES-LLM (Tree Understanding and Reasoning Enhanced System)
> 基于多模态大语言模型的城市街道树木参数理解与推理

---

## 项目结构

```
src/
├── data/                    # 数据加载与预处理
│   ├── mls_dataset.py       # MLS树木数据集
│   └── preprocessing.py     # 点云预处理（地面滤除、分割、参数提取）
│
├── models/                  # 核心模型
│   ├── tree_voxel_tokenizer.py   # TreeVoxel Tokenizer（树木3D分词器）
│   ├── tree_part_attention.py    # TreePartAttention（部位感知注意力）
│   ├── projection_layer.py       # 点云→LLM语义投影层
│   └── trees_llm.py              # TREES-LLM主模型
│
├── scripts/                 # 工具脚本
│   ├── preprocess_mls_data.py    # MLS数据预处理
│   ├── train_tokenizer.py        # Tokenizer预训练
│   ├── inference.py              # 推理脚本
│   └── eval_metrics.py           # 评估指标
│
├── configs/                 # 配置文件
│   ├── experiment_baseline.yaml  # 基线实验配置
│   └── inference.yaml            # 推理配置
│
├── api/                     # FastAPI后端
│   ├── main.py              # API入口
│   ├── schemas.py           # 数据模型
│   └── services.py          # 业务逻辑
│
├── utils/                   # 工具函数
│   └── visualization.py     # 可视化
│
└── notebooks/               # Jupyter分析笔记
    ├── 01_exploration/      # 数据探索
    ├── 02_preprocessing/    # 预处理实验
    ├── 03_model_development/# 模型开发
    └── 04_analysis/         # 结果分析
```

---

## 快速开始

### 1. 环境安装

```bash
# 创建conda环境
conda env create -f environment.yml
conda activate trees-llm

# 或手动安装
pip install -r requirements.txt
```

### 2. 数据预处理

```bash
# 预处理原始MLS数据
python scripts/preprocess_mls_data.py \
    --input data/00_raw \
    --output data/01_preprocessed
```

### 3. 快速推理（无需训练，使用Claude API）

```bash
# 提取参数
python scripts/inference.py --input data/00_raw/sample.las

# 问答模式
python scripts/inference.py \
    --input data/00_raw/sample.las \
    --question "这棵树长得健不健康？"

# 生成报告
python scripts/inference.py \
    --input data/00_raw/sample.las \
    --report
```

### 4. 训练（可选）

```bash
# Stage 1: 预训练 TreeVoxel Tokenizer
python scripts/train_tokenizer.py \
    --config configs/experiment_baseline.yaml
```

### 5. 启动API服务

```bash
python api/main.py

# 访问 http://localhost:8000/docs 查看API文档
```

---

## 核心模块说明

### TreeVoxel Tokenizer

将MLS点云离散化为树木语义token序列。

```python
from src.models.tree_voxel_tokenizer import TreeVoxelTokenizer

tokenizer = TreeVoxelTokenizer(
    num_geo_tokens=2048,    # 几何码本大小
    num_sem_tokens=2048,    # 语义码本大小
    token_dim=128,          # token维度
    trunk_voxel_size=0.02,  # 树干体素2cm
    crown_voxel_size=0.10,  # 树冠体素10cm
)

outputs = tokenizer(points)  # points: (B, N, 3)
# outputs["geo_tokens"]: (B, N) 几何token索引
# outputs["sem_tokens"]: (B, N) 语义token索引
```

### TREES-LLM

树木领域大语言模型。

```python
from src.models.trees_llm import TREESLLM, TREESLLMConfig

config = TREESLLMConfig(
    base_model="meta-llama/Meta-Llama-3-8B-Instruct",
    use_lora=True,
)

model = TREESLLM(config)

# 生成回答
answer = model.generate(point_embeds, "这棵树有多高？")
```

---

## API接口

| 端点 | 方法 | 描述 |
|------|------|------|
| `/extract` | POST | 上传点云，提取树木参数 |
| `/ask` | POST | 智能问答 |
| `/report` | POST | 生成调查报告 |
| `/scene-graph` | POST | 构建空间关系图 |

---

## 依赖

- Python 3.10+
- PyTorch 2.0+
- Open3D
- laspy
- transformers
- FastAPI
- Claude API Key (可选，用于问答和报告生成)

---

## 许可证

MIT
